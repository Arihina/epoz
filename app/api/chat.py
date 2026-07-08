from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import iterate_in_threadpool

from app.db.database import get_db, AsyncSessionLocal
from app.db import crud

from app.core.llm import stream_answer
from app.core.auth import get_user_id

from app.api.deps import get_owned_session

router = APIRouter()

_histories: dict[UUID, list[tuple[str, str]]] = {}


def _fmt_session(s) -> dict:
    return {"id": s.id, "title": s.title,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat()}


def _fmt_message(m) -> dict:
    sources = m.sources if m.sources else []
    fb = m.feedback
    return {
        "id": m.id, "role": m.role, "content": m.content, "sources": sources,
        "created_at": m.created_at.isoformat(),
        "feedback": {"vote": fb.vote, "comment": fb.comment} if fb else None,
    }


@router.post("/sessions", status_code=201)
async def create_session(
    body: dict = Body(default={}),
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db),
):
    s = await crud.create_session(db, user_id=user_id, title=body.get("title"))
    return _fmt_session(s)


@router.get("/sessions")
async def list_sessions(
    user_id: UUID = Depends(get_user_id),
    db: AsyncSession = Depends(get_db)
):
    return [_fmt_session(s) for s in await crud.list_sessions(db, user_id)]


@router.get("/sessions/{session_id}/messages")
async def get_messages(
    s=Depends(get_owned_session),
    db: AsyncSession = Depends(get_db),
):
    return [_fmt_message(m) for m in await crud.get_messages(db, s.id)]


@router.patch("/sessions/{session_id}")
async def rename_session(
    body: dict = Body(...),
    s=Depends(get_owned_session),
    db: AsyncSession = Depends(get_db),
):
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(422, "title не может быть пустым")
    return _fmt_session(await crud.rename_session(db, s, title))


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    s=Depends(get_owned_session),
    db: AsyncSession = Depends(get_db),
):
    await crud.delete_session(db, s)
    _histories.pop(s.id, None)


@router.post("/sessions/{session_id}/chat")
async def chat(
    body: dict = Body(...),
    s=Depends(get_owned_session),
    db: AsyncSession = Depends(get_db),
):
    question: str = body.get("message", "").replace("?", "").strip()
    if not question:
        raise HTTPException(422, "Пустой вопрос")

    if s.title is None:
        await crud.rename_session(
            db, s,
            question[:80] + ("…" if len(question) > 80 else "")
        )

    history = _histories.setdefault(
        s.id, await crud.build_history(db, s.id)
    )

    async def _save(full_answer: str, sources: list[str] | None):
        async with AsyncSessionLocal() as write_db:
            await crud.add_message(write_db, s.id, "user", question)
            msg = await crud.add_message(
                write_db, s.id, "assistant", full_answer, sources
            )
            return msg.id

    state: dict = {"answer": "", "sources": None, "message_id": None}

    def _gen():
        full_answer = ""
        sources: list[str] | None = None
        first = True

        for token, used, chunks in stream_answer(question, history):
            if first:
                first = False
                if chunks:
                    chunks_payload = [
                        {"text": c["text"], "source": c["source"],
                         "score": round(c["score"], 3)}
                        for c in chunks
                    ]
                    yield f"data: {json.dumps({'chunks': chunks_payload}, ensure_ascii=False)}\n\n"
                continue

            full_answer += token
            sources = used
            if token:
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        if sources:
            src_text = "\n\nИсточники:\n" + \
                "\n".join(f"- {src}" for src in sources)
            yield f"data: {json.dumps({'token': src_text}, ensure_ascii=False)}\n\n"
            full_answer += src_text

        state["answer"] = full_answer
        state["sources"] = sources

    async def _async_gen():
        async for chunk in iterate_in_threadpool(_gen()):
            yield chunk

        message_id = await _save(state["answer"], state["sources"])
        yield f"data: {json.dumps({'message_id': str(message_id)}, ensure_ascii=False)}\n\n"

        history.append(("user", question))
        history.append(("assistant", state["answer"]))
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _async_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions/{session_id}/reset")
async def reset(s=Depends(get_owned_session)):
    _histories.pop(s.id, None)
    return {"status": "ok"}


@router.get("/sessions/clear")
async def clear_all_sessions(
    db: AsyncSession = Depends(get_db),
):
    await crud.delete_all_sessions(db)
    _histories.clear()

    return {"status": "ok", "message": "Все чаты удалены"}
