from __future__ import annotations

import json
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.concurrency import iterate_in_threadpool

from app.db.database import get_db, SessionLocal
from app.db import crud
from app.core.llm import stream_answer

router = APIRouter()


_histories: dict[int, list[tuple[str, str]]] = {}


def _fmt_session(s) -> dict:
    return {"id": s.id, "title": s.title,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat()}


def _fmt_message(m) -> dict:
    sources = json.loads(m.sources) if m.sources else []
    fb = m.feedback
    return {
        "id": m.id, "role": m.role, "content": m.content, "sources": sources,
        "created_at": m.created_at.isoformat(),
        "feedback": {"vote": fb.vote, "comment": fb.comment} if fb else None,
    }


@router.post("/sessions", status_code=201)
def create_session(body: dict = Body(default={}), db: Session = Depends(get_db)):
    s = crud.create_session(db, title=body.get("title"))
    return _fmt_session(s)


@router.get("/sessions")
def list_sessions(db: Session = Depends(get_db)):
    return [_fmt_session(s) for s in crud.list_sessions(db)]


@router.get("/sessions/{session_id}/messages")
def get_messages(session_id: int, db: Session = Depends(get_db)):
    if not crud.get_session(db, session_id):
        raise HTTPException(404, "Чат не найден")
    return [_fmt_message(m) for m in crud.get_messages(db, session_id)]


@router.patch("/sessions/{session_id}")
def rename_session(session_id: int, body: dict = Body(...), db: Session = Depends(get_db)):
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(422, "title не может быть пустым")
    s = crud.rename_session(db, session_id, title)
    if not s:
        raise HTTPException(404, "Чат не найден")
    return _fmt_session(s)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: int, db: Session = Depends(get_db)):
    if not crud.delete_session(db, session_id):
        raise HTTPException(404, "Чат не найден")
    _histories.pop(session_id, None)


@router.post("/sessions/{session_id}/chat")
def chat(session_id: int, body: dict = Body(...), db: Session = Depends(get_db)):
    s = crud.get_session(db, session_id)
    if not s:
        raise HTTPException(404, "Чат не найден")

    question: str = body.get("message", "").replace("?", "").strip()
    if not question:
        raise HTTPException(422, "Пустой вопрос")

    if s.title is None:
        db.query(type(s)).filter_by(id=session_id).update(
            {"title": question[:80] + ("…" if len(question) > 80 else "")}
        )
        db.commit()

    history = _histories.setdefault(
        session_id, crud.build_history(db, session_id))

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
                "\n".join(f"- {s}" for s in sources)
            yield f"data: {json.dumps({'token': src_text}, ensure_ascii=False)}\n\n"
            full_answer += src_text

        write_db = SessionLocal()
        try:
            crud.add_message(write_db, session_id, "user", question)
            msg = crud.add_message(write_db, session_id,
                                   "assistant", full_answer, sources)
            yield f"data: {json.dumps({'message_id': msg.id}, ensure_ascii=False)}\n\n"
        finally:
            write_db.close()

        history.append(("user", question))
        history.append(("assistant", full_answer))
        yield "data: [DONE]\n\n"

    async def _async_gen():
        async for chunk in iterate_in_threadpool(_gen()):
            yield chunk

    return StreamingResponse(
        _async_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions/{session_id}/reset")
def reset(session_id: int):
    _histories.pop(session_id, None)
    return {"status": "ok"}
