from __future__ import annotations
from typing import Generator
import ollama

from app.core.retrieval import retrieve, DocResult


OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma2:2b"
RAG_MIN_SCORE = 0.013


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


def _general_prompt(history: list[tuple[str, str]], question: str) -> str:
    return f"""Ты русскоязычный AI ассистент.
Отвечай ТОЛЬКО на русском языке.
Это общий вопрос, не связанный с документами.
Отвечай свободно, как обычный ассистент.
Источники указывать НЕ НУЖНО.

Предыдущий диалог:
{_format_history(history)}

Вопрос пользователя:
{question}

Ответ:""".strip()


def _rag_prompt(
    docs: list[DocResult],
    history: list[tuple[str, str]],
    question: str,
) -> str:
    context = "\n\n".join(
        f"[Источник {i}]\n{d['text']}" for i, d in enumerate(docs, 1)
    )
    return f"""Ты русскоязычный AI ассистент.
Отвечай ТОЛЬКО на русском языке.
Используй ТОЛЬКО ту информацию из контекста, которая действительно нужна для ответа.
Если информация не использовалась — НЕ УПОМИНАЙ источник.

Контекст:
{context}

Предыдущий диалог:
{_format_history(history)}

Вопрос пользователя:
{question}

Ответ:""".strip()


def find_used_sources(answer: str, docs: list[DocResult]) -> list[str]:
    lower = answer.lower()
    used: set[str] = set()
    for doc in docs:
        overlap = sum(1 for w in doc["text"].lower().split() if w in lower)
        if overlap > 5:
            used.add(doc["source"])
    return sorted(used)


def stream_answer(
    question: str,
    history: list[tuple[str, str]],
) -> Generator[tuple[str, list[str] | None, list[DocResult] | None], None, None]:
    if is_small_talk(question):
        prompt = _general_prompt(history, question)
        docs = None
    else:
        docs = retrieve(question)
        if not docs or docs[0]["score"] < RAG_MIN_SCORE:
            prompt = _general_prompt(history, question)
            docs = None
        else:
            prompt = _rag_prompt(docs, history, question)

    yield "", None, docs

    stream = _client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    full_answer = ""
    for chunk in stream:
        token: str = chunk["message"]["content"]
        full_answer += token
        yield token, None, None

    used_sources: list[str] | None = None
    if docs:
        used = find_used_sources(full_answer, docs)
        if used:
            used_sources = used
    yield "", used_sources, None
