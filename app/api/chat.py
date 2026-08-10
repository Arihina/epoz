from __future__ import annotations

import json
import time
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.db.database import get_db
from app.db import crud

from app.core.llm import stream_answer
from app.core.auth import get_user_id
from app.api.content import CHAT_TEXT_TYPES, extract_text
from app.api.deps import get_owned_completion
from app.api.generation import (
    collect, format_sources, persist, sampling_options,
)

router = APIRouter(prefix="/v1/chat/completions", tags=["chat"])

DEFAULT_MODEL = "epoz"

_INSTRUCTION_ROLES = ("system", "developer")
_DIALOG_ROLES = ("user", "assistant")


def _extract(body: dict) -> dict:
    model = body.get("model") or DEFAULT_MODEL
    messages = body.get("messages")
    stream = bool(body.get("stream", False))

    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "messages обязателен и не должен быть пустым")

    n = body.get("n")
    if n is not None and n != 1:
        raise HTTPException(
            400, "Поддерживается только n=1: сервис возвращает один вариант ответа")

    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        raise HTTPException(
            400, 'последнее сообщение должно иметь role="user"')

    question = extract_text(last.get("content", ""), CHAT_TEXT_TYPES).strip()
    if not question:
        raise HTTPException(400, "Пустой вопрос")

    instructions = "\n".join(
        text for text in (
            extract_text(m.get("content", ""), CHAT_TEXT_TYPES)
            for m in messages
            if isinstance(m, dict) and m.get("role") in _INSTRUCTION_ROLES
        ) if text.strip()
    ) or None

    history = [
        (m["role"], extract_text(m.get("content", ""), CHAT_TEXT_TYPES))
        for m in messages[:-1]
        if isinstance(m, dict) and m.get("role") in _DIALOG_ROLES
    ]

    stream_options = body.get("stream_options") or {}
    if not isinstance(stream_options, dict):
        raise HTTPException(400, "stream_options должен быть объектом")

    return {
        "model": model,
        "history": history,
        "question": question,
        "instructions": instructions,
        "stream": stream,
        "include_usage": bool(stream_options.get("include_usage", False)),
        "conversation_id_raw": body.get("conversation_id"),
        "options": sampling_options(
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_completion_tokens",
                                body.get("max_tokens")),
        ),
    }


def _completion_object(
    completion_id: str, created: int, model: str, conversation_id_str: str | None,
    content: str, usage: dict,
) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "conversation_id": conversation_id_str,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "refusal": None,
                "annotations": [],
            },
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": usage,
    }


def _chunk(
    completion_id: str, created: int, model: str, conversation_id_str: str | None,
    delta: dict | None, finish_reason: str | None = None, usage: dict | None = None,
) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "conversation_id": conversation_id_str,
        "choices": [] if delta is None else [{
            "index": 0,
            "delta": delta,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
    }
    if usage is not None:
        payload["usage"] = usage

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("")
async def chat_completions(
    body: dict = Body(...),
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    req = _extract(body)
    model = req["model"]

    conversation_id: UUID | None = None
    if req["conversation_id_raw"] is not None:
        try:
            conversation_id = UUID(str(req["conversation_id_raw"]))
        except ValueError:
            raise HTTPException(400, "conversation_id должен быть UUID")
        if await crud.get_conversation(db, conversation_id, user_id) is None:
            raise HTTPException(404, "Чат (conversation_id) не найден")

    assistant_id = uuid4()
    completion_id = f"chatcmpl-{assistant_id}"
    created = int(time.time())
    conversation_id_str = str(conversation_id) if conversation_id else None

    if not req["stream"]:
        gen = await run_in_threadpool(
            collect, req["question"], req["history"],
            req["instructions"], req["options"],
        )
        answer = gen.answer + format_sources(gen.used_sources)

        await persist(user_id, req["question"], answer, gen.used_sources,
                      gen.docs, model, assistant_id, conversation_id, gen.usage)

        return _completion_object(completion_id, created, model,
                                  conversation_id_str, answer, gen.usage)

    state = {"answer": "", "used_sources": None, "docs": None, "usage": None}

    def _gen():
        first = True
        for token, used, docs, usage in stream_answer(
            req["question"], req["history"],
            instructions=req["instructions"], options=req["options"],
        ):
            if first:
                first = False
                state["docs"] = docs
                yield _chunk(completion_id, created, model, conversation_id_str,
                             {"role": "assistant", "content": ""})
                continue

            if token:
                state["answer"] += token
                yield _chunk(completion_id, created, model, conversation_id_str,
                             {"content": token})

            if used is not None:
                state["used_sources"] = used
            if usage is not None:
                state["usage"] = usage

        src_text = format_sources(state["used_sources"])
        if src_text:
            state["answer"] += src_text
            yield _chunk(completion_id, created, model, conversation_id_str,
                         {"content": src_text})

        yield _chunk(completion_id, created, model, conversation_id_str,
                     {}, finish_reason="stop")

        if req["include_usage"]:
            yield _chunk(completion_id, created, model, conversation_id_str,
                         None, usage=state["usage"] or {
                             "prompt_tokens": 0, "completion_tokens": 0,
                             "total_tokens": 0})

    async def _async_gen():
        async for chunk in iterate_in_threadpool(_gen()):
            yield chunk
        yield "data: [DONE]\n\n"

        await persist(
            user_id, req["question"], state["answer"], state["used_sources"],
            state["docs"], model, assistant_id, conversation_id, state["usage"],
        )

    return StreamingResponse(
        _async_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{completion_id}")
async def get_completion(msg=Depends(get_owned_completion)):
    if msg.role != "assistant":
        raise HTTPException(404, "Completion не найден")

    prompt_tokens = msg.prompt_tokens or 0
    completion_tokens = msg.completion_tokens or 0

    return _completion_object(
        f"chatcmpl-{msg.id}", int(msg.created_at.timestamp()), msg.model,
        str(msg.conversation_id) if msg.conversation_id else None,
        msg.content,
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


@router.delete("/{completion_id}")
async def delete_completion(
    msg=Depends(get_owned_completion),
    db: AsyncSession = Depends(get_db),
):
    if msg.role != "assistant":
        raise HTTPException(404, "Completion не найден")

    completion_id = f"chatcmpl-{msg.id}"
    await crud.delete_message(db, msg)

    return {"id": completion_id, "object": "chat.completion.deleted", "deleted": True}
