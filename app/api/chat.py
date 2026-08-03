from __future__ import annotations

import json
import time
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.db.database import get_db, AsyncSessionLocal
from app.db import crud

from app.core.llm import stream_answer
from app.core.auth import get_user_id
from app.core.retrieval import DocResult
from app.api.deps import get_owned_completion

router = APIRouter(prefix="/v1/chat/completions", tags=["chat"])

DEFAULT_MODEL = "epoz"
TITLE_MAX_LEN = 80


def _trim_docs(docs: list[DocResult] | None) -> list[dict] | None:
    if not docs:
        return None
    return [
        {"text": d["text"], "source": d["source"],
            "score": round(d["score"], 3)}
        for d in docs
    ]


def _extract(body: dict) -> tuple[str, list[tuple[str, str]], str, bool, str | None]:
    model = body.get("model") or DEFAULT_MODEL
    messages = body.get("messages")
    stream = body.get("stream", False)
    conversation_id_raw = body.get("conversation_id")

    if not isinstance(messages, list) or not messages:
        raise HTTPException(422, "messages обязателен и не должен быть пустым")

    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        raise HTTPException(
            422, 'последнее сообщение должно иметь role="user"')

    question = str(last.get("content", "")).replace("?", "").strip()
    if not question:
        raise HTTPException(422, "Пустой вопрос")

    history = [
        (m["role"], m["content"])
        for m in messages[:-1]
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ]

    return model, history, question, bool(stream), conversation_id_raw


def _collect(question: str, history: list[tuple[str, str]]):
    full_answer = ""
    used_sources: list[str] | None = None
    docs = None
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for token, used, d, u in stream_answer(question, history):
        if d is not None:
            docs = d
        if token:
            full_answer += token
        if used is not None:
            used_sources = used
        if u is not None:
            usage = u

    return full_answer, used_sources, docs, usage


async def _persist(
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
            sources=used_sources, retrieved_chunks=_trim_docs(docs), model=model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            id=assistant_id, conversation_id=conversation_id,
        )
        if conversation_id is not None:
            title = question[:TITLE_MAX_LEN] + \
                ("…" if len(question) > TITLE_MAX_LEN else "")
            await crud.touch_conversation(write_db, conversation_id, title=title)


@router.post("")
async def chat_completions(
    body: dict = Body(...),
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    model, history, question, stream, conversation_id_raw = _extract(body)

    conversation_id: UUID | None = None
    if conversation_id_raw is not None:
        try:
            conversation_id = UUID(str(conversation_id_raw))
        except ValueError:
            raise HTTPException(422, "conversation_id должен быть UUID")
        if await crud.get_conversation(db, conversation_id, user_id) is None:
            raise HTTPException(404, "Чат (conversation_id) не найден")

    assistant_id = uuid4()
    completion_id = f"chatcmpl-{assistant_id}"
    created = int(time.time())
    conversation_id_str = str(conversation_id) if conversation_id else None

    if not stream:
        full_answer, used_sources, docs, usage = await run_in_threadpool(
            _collect, question, history
        )
        if used_sources:
            full_answer += "\n\nИсточники:\n" + \
                "\n".join(f"- {s}" for s in used_sources)

        await _persist(user_id, question, full_answer, used_sources, docs, model,
                       assistant_id, conversation_id, usage)

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "conversation_id": conversation_id_str,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_answer},
                "finish_reason": "stop",
            }],
            "usage": usage,
        }

    state = {"answer": "", "used_sources": None, "docs": None, "usage": None}

    def _gen():
        first = True
        for token, used, docs, usage in stream_answer(question, history):
            if first:
                first = False
                state["docs"] = docs
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "conversation_id": conversation_id_str,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                continue

            if token:
                state["answer"] += token
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "conversation_id": conversation_id_str,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            if used is not None:
                state["used_sources"] = used
            if usage is not None:
                state["usage"] = usage

        if state["used_sources"]:
            src_text = "\n\nИсточники:\n" + \
                "\n".join(f"- {s}" for s in state["used_sources"])
            state["answer"] += src_text
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "conversation_id": conversation_id_str,
                "choices": [{"index": 0, "delta": {"content": src_text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        final_chunk = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "conversation_id": conversation_id_str,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

    async def _async_gen():
        async for chunk in iterate_in_threadpool(_gen()):
            yield chunk
        yield "data: [DONE]\n\n"

        await _persist(
            user_id, question, state["answer"], state["used_sources"],
            state["docs"], model, assistant_id, conversation_id, state["usage"],
        )

    return StreamingResponse(
        _async_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{completion_id}")
async def get_completion(completion_id: str, msg=Depends(get_owned_completion)):
    if msg.role != "assistant":
        raise HTTPException(404, "Completion не найден")

    prompt_tokens = msg.prompt_tokens or 0
    completion_tokens = msg.completion_tokens or 0

    return {
        "id": f"chatcmpl-{msg.id}",
        "object": "chat.completion",
        "created": int(msg.created_at.timestamp()),
        "model": msg.model,
        "conversation_id": str(msg.conversation_id) if msg.conversation_id else None,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": msg.content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
