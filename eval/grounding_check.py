"""Проверка grounding: как модель отвечает, есть ли ответ в базе, и что в «Источниках».

Запуск (из корня репозитория):
    python eval/grounding_check.py

Требует: поднятый Qdrant с проиндексированной коллекцией, Ollama, модели в кэше.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm import stream_answer  # noqa: E402

QUERIES = [
    "что такое НМЦ",                                          # глоссарий — должен ответить и сослаться
    "минимальный срок подачи заявок на участие в закупке",     # есть в положении — должен ответить с источником
    "какой штраф за опоздание на заседание закупочной комиссии",  # НЕТ в положении — должен честно сказать «не найдено»
    "требования к поставщикам офисной мебели",                 # НЕТ в положении — должен честно сказать «не найдено»
]


def ask(question: str) -> tuple[str, list[str] | None, list | None]:
    answer = ""
    used: list[str] | None = None
    docs = None
    for token, used_src, d, _usage in stream_answer(question, []):
        if d is not None:
            docs = d
        if token:
            answer += token
        if used_src is not None:
            used = used_src
    return answer, used, docs


def main() -> None:
    for q in QUERIES:
        answer, used, docs = ask(q)
        print("=" * 78)
        print("ВОПРОС :", q)
        print(f"НАЙДЕНО: {len(docs) if docs else 0} чанков")
        print("ОТВЕТ  :", answer.strip())
        print("ИСТОЧНИКИ:", ", ".join(used) if used else "—")
        print()


if __name__ == "__main__":
    main()
