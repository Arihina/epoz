from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from fastapi import HTTPException

from app.core.llm import stream_answer
from app.core.retrieval import DocResult
from app.db.database import AsyncSessionLocal
from app.db import crud

TITLE_MAX_LEN = 80
SOURCES_HEADER = "\n\nПроанализированные источники:\n"


def format_sources(used_sources: list[str] | None) -> str:
    if not used_sources:
        return ""
    return SOURCES_HEADER + "\n".join(f"- {s}" for s in used_sources)


def trim_docs(docs: list[DocResult] | None) -> list[dict] | None:
    if not docs:
        return None
    return [
        {"text": d["text"], "source": d["source"],
            "score": round(d["score"], 3)}
        for d in docs
    ]


@dataclass
class Generation:
    answer: str = ""
    used_sources: list[str] | None = None
    docs: list[DocResult] | None = None
    usage: dict = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


def collect(question, history, instructions=None, options=None) -> Generation:
    out = Generation()

    for token, used, docs, usage in stream_answer(
        question, history, instructions=instructions, options=options
    ):
        if docs is not None:
            out.docs = docs
        if token:
            out.answer += token
        if used is not None:
            out.used_sources = used
        if usage is not None:
            out.usage = usage

    return out


def sampling_options(
    temperature=None, top_p=None, max_tokens=None
) -> dict | None:
    options: dict = {}

    if temperature is not None:
        options["temperature"] = _positive_number("temperature", temperature)
    if top_p is not None:
        options["top_p"] = _positive_number("top_p", top_p)
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise HTTPException(
                400, "max_tokens должен быть целым числом >= 1")
        options["num_predict"] = max_tokens

    return options or None


def _positive_number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(400, f"{name} должен быть числом")
    return float(value)


async def persist(
    user_id: UUID,
    question: str,
    answer: str,
    used_sources: list[str] | None,
    docs: list[DocResult] | None,
    model: str,
    assistant_id: UUID,
    conversation_id: UUID | None,
    usage: dict | None,
) -> None:
    usage = usage or {}

    async with AsyncSessionLocal() as write_db:
        await crud.add_message(
            write_db, user_id=user_id, role="user", content=question,
            conversation_id=conversation_id,
        )
        await crud.add_message(
            write_db, user_id=user_id, role="assistant", content=answer,
            sources=used_sources, retrieved_chunks=trim_docs(docs), model=model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            id=assistant_id, conversation_id=conversation_id,
        )
        if conversation_id is not None:
            title = question[:TITLE_MAX_LEN] + \
                ("…" if len(question) > TITLE_MAX_LEN else "")
            await crud.touch_conversation(write_db, conversation_id, title=title)
