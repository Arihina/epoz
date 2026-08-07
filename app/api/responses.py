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
from app.core.config import settings
from app.core.retrieval import DocResult
from app.api.deps import get_owned_completion

router = APIRouter(prefix="/v1/responses", tags=["responses"])

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


def _parse_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", "")) for p in content
            if isinstance(p, dict) and p.get("type") in ("input_text", "output_text")
        )
    return ""


def _extract(body: dict) -> tuple[str, list[tuple[str, str]], str, bool, str | None]:
    model = body.get("model") or DEFAULT_MODEL
    input_data = body.get("input")
    stream = body.get("stream", False)
    conversation_id_raw = body.get("conversation_id")

    if input_data is None:
        raise HTTPException(422, "input обязателен")

    if isinstance(input_data, str):
        question = input_data.replace("?", "").strip()
        if not question:
            raise HTTPException(422, "Пустой input")
        return model, [], question, bool(stream), conversation_id_raw

    if not isinstance(input_data, list) or not input_data:
        raise HTTPException(
            422, "input должен быть строкой или непустым списком items")

    last = input_data[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        raise HTTPException(
            422, 'последний item в input должен иметь role="user"')

    question = _parse_content(last.get("content")).replace("?", "").strip()
    if not question:
        raise HTTPException(422, "Пустой вопрос")

    history = [
        (m["role"], _parse_content(m.get("content")))
        for m in input_data[:-1]
        if isinstance(m, dict) and m.get("type", "message") == "message"
        and m.get("role") in ("user", "assistant")
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


def _usage_out(usage: dict) -> dict:
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _message_item(item_id: str, text: str, status: str) -> dict:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": text, "annotations": []},
        ],
    }


def _response_object(
    response_id: str, created: int, model: str, conversation_id_str: str | None,
    status: str, output: list[dict], usage: dict | None = None,
) -> dict:
    obj = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "conversation_id": conversation_id_str,
        "output": output,
    }
    if usage is not None:
        obj["usage"] = usage
    return obj


def _sse_event(seq: int, event_type: str, **fields) -> str:
    payload = {"type": event_type, "sequence_number": seq, **fields}
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("")
async def create_response(
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

        if history:
            raise HTTPException(
                422,
                "При переданном conversation_id input должен содержать "
                "только новый ход (без истории) — история собирается "
                "агентом из БД по conversation_id",
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

    if not stream:
        full_answer, used_sources, docs, usage = await run_in_threadpool(
            _collect, question, history
        )
        if used_sources:
            full_answer += "\n\nПроанализированные источники:\n" + \
                "\n".join(f"- {s}" for s in used_sources)

        await _persist(user_id, question, full_answer, used_sources, docs, model,
                       assistant_id, conversation_id, usage)

        return _response_object(
            response_id, created, model, conversation_id_str, "completed",
            output=[_message_item(item_id, full_answer, "completed")],
            usage=_usage_out(usage),
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
                "in_progress", output=[]),
        )
        yield _sse_event(
            _next(), "response.output_item.added",
            output_index=0, item=_message_item(item_id, "", "in_progress"),
        )
        yield _sse_event(
            _next(), "response.content_part.added",
            item_id=item_id, output_index=0, content_index=0,
            part={"type": "output_text", "text": "", "annotations": []},
        )

        first = True
        for token, used, docs, usage in stream_answer(question, history):
            if first:
                first = False
                state["docs"] = docs
                continue

            if token:
                state["answer"] += token
                yield _sse_event(
                    _next(), "response.output_text.delta",
                    item_id=item_id, output_index=0, content_index=0,
                    delta=token,
                )

            if used is not None:
                state["used_sources"] = used
            if usage is not None:
                state["usage"] = usage

        if state["used_sources"]:
            src_text = "\n\nПроанализированные источники:\n" + \
                "\n".join(f"- {s}" for s in state["used_sources"])
            state["answer"] += src_text
            yield _sse_event(
                _next(), "response.output_text.delta",
                item_id=item_id, output_index=0, content_index=0,
                delta=src_text,
            )

        yield _sse_event(
            _next(), "response.output_text.done",
            item_id=item_id, output_index=0, content_index=0,
            text=state["answer"],
        )
        yield _sse_event(
            _next(), "response.content_part.done",
            item_id=item_id, output_index=0, content_index=0,
            part={"type": "output_text",
                  "text": state["answer"], "annotations": []},
        )

        final_item = _message_item(item_id, state["answer"], "completed")
        yield _sse_event(_next(), "response.output_item.done",
                         output_index=0, item=final_item)

        usage_out = _usage_out(state["usage"] or {})
        yield _sse_event(
            _next(), "response.completed",
            response=_response_object(
                response_id, created, model, conversation_id_str,
                "completed", output=[final_item], usage=usage_out),
        )

    async def _async_gen():
        try:
            async for chunk in iterate_in_threadpool(_gen()):
                yield chunk
        except Exception as e:
            yield _sse_event(9999, "error", message=str(e), code=None, param=None)
            return

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
async def get_response(msg=Depends(get_owned_completion)):
    if msg.role != "assistant":
        raise HTTPException(404, "Response не найден")

    usage_out = {
        "input_tokens": msg.prompt_tokens or 0,
        "output_tokens": msg.completion_tokens or 0,
        "total_tokens": (msg.prompt_tokens or 0) + (msg.completion_tokens or 0),
    }

    return _response_object(
        f"resp_{msg.id}", int(msg.created_at.timestamp()), msg.model,
        str(msg.conversation_id) if msg.conversation_id else None,
        "completed",
        output=[_message_item(f"msg_{msg.id}", msg.content, "completed")],
        usage=usage_out,
    )
