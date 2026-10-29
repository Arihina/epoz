from __future__ import annotations
from typing import Generator
import re

import ollama

from app.core.retrieval import retrieve, lookup_glossary, expand_query, DocResult
from app.core.config import settings


OLLAMA_HOST = settings.OLLAMA_HOST
OLLAMA_MODEL = settings.OLLAMA_MODEL


_client = ollama.Client(host=OLLAMA_HOST)


_SMALL_TALK_PATTERNS = (
    "привет", "здравств", "как тебя", "кто ты",
    "кто я", "как меня зовут", "меня зовут",
    "hello", "hi",
)


def is_small_talk(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _SMALL_TALK_PATTERNS)


def _format_history(history: list[tuple[str, str]], max_turns: int = 10) -> str:
    recent = history[-(max_turns * 2):]
    lines = []
    for role, text in recent:
        label = "Пользователь" if role == "user" else "Ассистент"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _instructions_block(instructions: str | None) -> str:
    if not instructions or not instructions.strip():
        return ""
    return f"\nДополнительные инструкции пользователя:\n{instructions.strip()}\n"


def _general_prompt(
    history: list[tuple[str, str]],
    question: str,
    instructions: str | None = None,
) -> str:
    return f"""Ты русскоязычный AI ассистент.
Отвечай ТОЛЬКО на русском языке.
Это общий вопрос, не связанный с документами.
Отвечай свободно, как обычный ассистент.
Источники указывать НЕ НУЖНО.
{_instructions_block(instructions)}
Предыдущий диалог:
{_format_history(history)}

Вопрос пользователя:
{question}

Ответ:""".strip()


def _rag_prompt(
    docs: list[DocResult],
    history: list[tuple[str, str]],
    question: str,
    instructions: str | None = None,
) -> str:
    if docs:
        context = "\n\n".join(
            f"[Источник {i}]\n{d['text']}" for i, d in enumerate(docs, 1)
        )
    else:
        context = "(в базе знаний по вопросу ничего не найдено)"
    return f"""Ты русскоязычный AI ассистент.
Отвечай ТОЛЬКО на русском языке.
Отвечай СТРОГО на основании приведённого контекста.
Если в контексте нет ответа на вопрос — прямо напиши, что в положении такая информация не найдена, и НЕ додумывай.
Не выдумывай нормы, сроки и критерии, которых нет в контексте.
{_instructions_block(instructions)}
Контекст:
{context}

Предыдущий диалог:
{_format_history(history)}

Вопрос пользователя:
{question}

Ответ:""".strip()


_STOPWORDS = frozenset({
    "и", "в", "во", "не", "на", "по", "с", "со", "к", "ко", "о", "об", "от", "до",
    "из", "за", "для", "при", "как", "что", "это", "этот", "эта", "или", "а", "но",
    "да", "же", "бы", "то", "так", "уже", "ещё", "если", "также", "который",
    "которая", "которое", "которые", "все", "весь", "вся", "всё", "более", "менее",
    "без", "под", "над", "у", "которых",
})


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def find_used_sources(answer: str, docs: list[DocResult]) -> list[str]:
    """Источники, реально использованные в ответе: пересечение слов ответа и чанка
    (без стоп-слов, порог по доле покрытия, а не по абсолютному числу слов)."""
    answer_words = {w for w in _tokenize(answer) if w not in _STOPWORDS and len(w) > 2}
    if not answer_words:
        return []

    scored: list[tuple[str, float]] = []
    for doc in docs:
        doc_words = [w for w in _tokenize(doc["text"]) if w not in _STOPWORDS and len(w) > 2]
        if not doc_words:
            continue
        overlap = len(answer_words & set(doc_words))
        fraction = overlap / len(doc_words)
        if overlap >= 3 and fraction >= 0.1:
            scored.append((doc["source"], fraction))

    used: list[str] = []
    seen: set[str] = set()
    for src, _ in sorted(scored, key=lambda x: -x[1]):
        if src not in seen:
            seen.add(src)
            used.append(src)
    return used


_REWRITE_PROMPT = """Ты готовишь поисковый запрос для поиска по Единому положению о закупках Госкорпорации «Ростех».
Учитывая диалог, перепиши последний вопрос пользователя в самостоятельный поисковый запрос:
- раскрой местоимения и эллипсис («это», «он», «а какие сроки») через контекст;
- замени аббревиатуры на расшифровки, где знаешь;
- верни ТОЛЬКО поисковый запрос, без пояснений и кавычек.

Диалог:
{history}

Последний вопрос:
{question}

Поисковый запрос:"""


def _rewrite_query(question: str, history: list[tuple[str, str]]) -> str:
    """Переписать вопрос в самостоятельный поисковый запрос с учётом истории."""
    if not history:
        return question
    prompt = _REWRITE_PROMPT.format(
        history=_format_history(history),
        question=question,
    )
    try:
        resp = _client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        rewritten = (resp.get("message", {}).get("content") or "").strip()
        return rewritten or question
    except Exception as exc:
        print(f"[llm] Ошибка переписывания запроса: {exc}")
        return question


def stream_answer(
    question: str,
    history: list[tuple[str, str]],
    instructions: str | None = None,
    options: dict | None = None,
) -> Generator[tuple[str, list[str] | None, list[DocResult] | None, dict | None], None, None]:
    if is_small_talk(question):
        prompt = _general_prompt(history, question, instructions)
        docs = None
    else:
        search_query = _rewrite_query(question, history) if settings.QUERY_REWRITE else question
        docs = retrieve(expand_query(search_query))
        glossary_hits = lookup_glossary(question)
        if glossary_hits:
            docs = glossary_hits + docs

        prompt = _rag_prompt(docs, history, question, instructions)

    yield "", None, docs, None

    stream = _client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options=options or None,
    )
    full_answer = ""
    prompt_tokens = 0
    completion_tokens = 0
    for chunk in stream:
        token: str = chunk["message"]["content"]
        full_answer += token
        if token:
            yield token, None, None, None
        if chunk.get("done"):
            prompt_tokens = chunk.get("prompt_eval_count", 0) or 0
            completion_tokens = chunk.get("eval_count", 0) or 0

    used_sources: list[str] | None = None
    if docs:
        used = find_used_sources(full_answer, docs)
        if used:
            used_sources = used

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    yield "", used_sources, None, usage
