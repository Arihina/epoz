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
from app.core.config import settings
from app.api.content import RESPONSES_TEXT_TYPES, extract_text
from app.api.deps import get_owned_completion, parse_completion_id
from app.api.generation import (
    collect, format_sources, persist, sampling_options,
)

router = APIRouter(prefix="/v1/responses", tags=["responses"])

DEFAULT_MODEL = "epoz"

_INSTRUCTION_ROLES = ("system", "developer")
_DIALOG_ROLES = ("user", "assistant")


def _conversation_field(body: dict):
    conversation = body.get("conversation")

    if isinstance(conversation, dict):
        return conversation.get("id")
    if conversation is not None:
        return conversation

    return body.get("conversation_id")


def _extract(body: dict) -> dict:
    model = body.get("model") or DEFAULT_MODEL
    input_data = body.get("input")
    stream = bool(body.get("stream", False))

    options = sampling_options(
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        max_tokens=body.get("max_output_tokens"),
    )

    common = {
        "model": model,
        "stream": stream,
        "options": options,
        "conversation_raw": _conversation_field(body),
        "previous_response_id": body.get("previous_response_id"),
        "store": bool(body.get("store", True)),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "max_output_tokens": body.get("max_output_tokens"),
        "metadata": body.get("metadata") or {},
        "instructions": body.get("instructions"),
    }

    if input_data is None:
        raise HTTPException(400, "input обязателен")

    if isinstance(input_data, str):
        question = input_data.strip()
        if not question:
            raise HTTPException(400, "Пустой input")
        return {**common, "history": [], "question": question}

    if not isinstance(input_data, list) or not input_data:
        raise HTTPException(
            400, "input должен быть строкой или непустым списком items")

    last = input_data[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        raise HTTPException(
            400, 'последний item в input должен иметь role="user"')

    question = extract_text(last.get("content"), RESPONSES_TEXT_TYPES).strip()
    if not question:
        raise HTTPException(400, "Пустой вопрос")

    items = [m for m in input_data[:-1]
             if isinstance(m, dict) and m.get("type", "message") == "message"]

    inline_instructions = "\n".join(
        text for text in (
            extract_text(m.get("content"), RESPONSES_TEXT_TYPES)
            for m in items if m.get("role") in _INSTRUCTION_ROLES
        ) if text.strip()
    )
    if inline_instructions:
        common["instructions"] = "\n".join(
            filter(None, [common["instructions"], inline_instructions]))

    history = [
        (m["role"], extract_text(m.get("content"), RESPONSES_TEXT_TYPES))
        for m in items if m.get("role") in _DIALOG_ROLES
    ]

    return {**common, "history": history, "question": question}


def _usage_out(usage: dict | None) -> dict:
    """input_tokens_details / output_tokens_details обязательны в объекте
    usage — без них официальный SDK не разбирает ответ."""
    usage = usage or {}
    pt = usage.get("prompt_tokens", 0) or 0
    ct = usage.get("completion_tokens", 0) or 0

    return {
        "input_tokens": pt,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": ct,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": usage.get("total_tokens", pt + ct),
    }


def _message_item(item_id: str, text: str, status: str) -> dict:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": text,
                "annotations": [], "logprobs": []},
        ],
    }


def _response_object(
    response_id: str, created: int, model: str, conversation_id_str: str | None,
    status: str, output: list[dict], usage: dict | None = None,
    req: dict | None = None, error: dict | None = None,
) -> dict:
    req = req or {}

    obj = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "error": error,
        "incomplete_details": None,
        "instructions": req.get("instructions"),
        "metadata": req.get("metadata") or {},
        "temperature": req.get("temperature"),
        "top_p": req.get("top_p"),
        "max_output_tokens": req.get("max_output_tokens"),
        "previous_response_id": req.get("previous_response_id"),
        "store": req.get("store", True),
        "truncation": "disabled",
        "text": {"format": {"type": "text"}},
        # Расширение платформы, не часть спецификации.
        "conversation_id": conversation_id_str,
    }
    if usage is not None:
        obj["usage"] = usage

    return obj


def _sse_event(seq: int, event_type: str, **fields) -> str:
    payload = {"type": event_type, "sequence_number": seq, **fields}
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _resolve_conversation(db, user_id, req) -> UUID | None:
    raw = req["conversation_raw"]

    if raw is None and req["previous_response_id"]:
        prev = await crud.get_message_for_user(
            db, parse_completion_id(req["previous_response_id"]), user_id)
        if prev is None:
            raise HTTPException(404, "previous_response_id не найден")
        return prev.conversation_id

    if raw is None:
        return None

    try:
        conversation_id = UUID(str(raw))
    except ValueError:
        raise HTTPException(400, "conversation должен быть UUID")

    if await crud.get_conversation(db, conversation_id, user_id) is None:
        raise HTTPException(404, "Чат (conversation) не найден")

    return conversation_id


@router.post("")
async def create_response(
    body: dict = Body(...),
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    req = _extract(body)
    model = req["model"]

    conversation_id = await _resolve_conversation(db, user_id, req)
    history = req["history"]

    if conversation_id is not None:
        if history:
            raise HTTPException(
                400,
                "При переданном conversation input должен содержать только "
                "новый ход (без истории) — история собирается агентом из БД",
            )
        history = [
            (m.role, m.content)
            for m in await crud.get_recent_conversation_messages(
                db, conversation_id, settings.HISTORY_LIMIT)
        ]

    assistant_id = uuid4()
    response_id = f"resp_{assistant_id}"
    item_id = f"msg_{assistant_id}"
    created = int(time.time())
    conversation_id_str = str(conversation_id) if conversation_id else None

    if not req["stream"]:
        gen = await run_in_threadpool(
            collect, req["question"], history, req["instructions"], req["options"],
        )
        answer = gen.answer + format_sources(gen.used_sources)

        if req["store"]:
            await persist(user_id, req["question"], answer, gen.used_sources,
                          gen.docs, model, assistant_id, conversation_id, gen.usage)

        return _response_object(
            response_id, created, model, conversation_id_str, "completed",
            output=[_message_item(item_id, answer, "completed")],
            usage=_usage_out(gen.usage), req=req,
        )

    state = {"answer": "", "used_sources": None, "docs": None, "usage": None}

    def _gen():
        seq = 0

        def _next():
            nonlocal seq
            seq += 1
            return seq

        yield _sse_event(
            _next(), "response.created",
            response=_response_object(
                response_id, created, model, conversation_id_str,
                "in_progress", output=[], req=req),
        )
        yield _sse_event(
            _next(), "response.in_progress",
            response=_response_object(
                response_id, created, model, conversation_id_str,
                "in_progress", output=[], req=req),
        )
        yield _sse_event(
            _next(), "response.output_item.added",
            output_index=0, item=_message_item(item_id, "", "in_progress"),
        )
        yield _sse_event(
            _next(), "response.content_part.added",
            item_id=item_id, output_index=0, content_index=0,
            part={"type": "output_text", "text": "",
                  "annotations": [], "logprobs": []},
        )

        first = True
        for token, used, docs, usage in stream_answer(
            req["question"], history,
            instructions=req["instructions"], options=req["options"],
        ):
            if first:
                first = False
                state["docs"] = docs
                continue

            if token:
                state["answer"] += token
                yield _sse_event(
                    _next(), "response.output_text.delta",
                    item_id=item_id, output_index=0, content_index=0,
                    delta=token, logprobs=[],
                )

            if used is not None:
                state["used_sources"] = used
            if usage is not None:
                state["usage"] = usage

        src_text = format_sources(state["used_sources"])
        if src_text:
            state["answer"] += src_text
            yield _sse_event(
                _next(), "response.output_text.delta",
                item_id=item_id, output_index=0, content_index=0,
                delta=src_text, logprobs=[],
            )

        yield _sse_event(
            _next(), "response.output_text.done",
            item_id=item_id, output_index=0, content_index=0,
            text=state["answer"], logprobs=[],
        )
        yield _sse_event(
            _next(), "response.content_part.done",
            item_id=item_id, output_index=0, content_index=0,
            part={"type": "output_text", "text": state["answer"],
                  "annotations": [], "logprobs": []},
        )

        final_item = _message_item(item_id, state["answer"], "completed")
        yield _sse_event(_next(), "response.output_item.done",
                         output_index=0, item=final_item)

        yield _sse_event(
            _next(), "response.completed",
            response=_response_object(
                response_id, created, model, conversation_id_str,
                "completed", output=[final_item],
                usage=_usage_out(state["usage"]), req=req),
        )

    async def _async_gen():
        seen = 0
        try:
            async for chunk in iterate_in_threadpool(_gen()):
                seen += 1
                yield chunk
        except Exception as e:
            yield _sse_event(seen + 1, "error",
                             message=str(e), code=None, param=None)
            return

        if req["store"]:
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
async def get_response(msg=Depends(get_owned_completion)):
    if msg.role != "assistant":
        raise HTTPException(404, "Response не найден")

    return _response_object(
        f"resp_{msg.id}", int(msg.created_at.timestamp()), msg.model,
        str(msg.conversation_id) if msg.conversation_id else None,
        "completed",
        output=[_message_item(f"msg_{msg.id}", msg.content, "completed")],
        usage=_usage_out({
            "prompt_tokens": msg.prompt_tokens or 0,
            "completion_tokens": msg.completion_tokens or 0,
        }),
    )


@router.delete("/{completion_id}")
async def delete_response(
    msg=Depends(get_owned_completion),
    db: AsyncSession = Depends(get_db),
):
    if msg.role != "assistant":
        raise HTTPException(404, "Response не найден")

    response_id = f"resp_{msg.id}"
    await crud.delete_message(db, msg)

    return {"id": response_id, "object": "response.deleted", "deleted": True}
