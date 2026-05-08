import json
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session

from app.models.models import ChatSession, ChatMessage, MessageFeedback


def create_session(db: Session, title: Optional[str] = None) -> ChatSession:
    s = ChatSession(title=title)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def get_session(db: Session, session_id: int) -> Optional[ChatSession]:
    return db.query(ChatSession).filter(ChatSession.id == session_id).first()


def list_sessions(db: Session) -> list[ChatSession]:
    return db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()


def rename_session(db: Session, session_id: int, title: str):
    s = get_session(db, session_id)
    if not s:
        return None
    s.title = title
    db.commit()
    db.refresh(s)
    return s


def delete_session(db: Session, session_id: int) -> bool:
    s = get_session(db, session_id)
    if not s:
        return False
    db.delete(s)
    db.commit()
    return True


def _touch(db: Session, session_id: int) -> None:
    db.query(ChatSession).filter(ChatSession.id == session_id).update(
        {"updated_at": datetime.now(timezone.utc)}
    )
    db.commit()


def add_message(db: Session, session_id: int, role: str,
                content: str, sources: Optional[list[str]] = None) -> ChatMessage:
    msg = ChatMessage(
        session_id=session_id, role=role, content=content,
        sources=json.dumps(sources, ensure_ascii=False) if sources else None,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    _touch(db, session_id)
    return msg


def get_messages(db: Session, session_id: int) -> list[ChatMessage]:
    return (db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
            .all())


def get_message(db: Session, message_id: int) -> Optional[ChatMessage]:
    return db.query(ChatMessage).filter(ChatMessage.id == message_id).first()


def build_history(db: Session, session_id: int) -> list[tuple[str, str]]:
    return [(m.role, m.content) for m in get_messages(db, session_id)]


def upsert_feedback(db: Session, message_id: int,
                    vote, comment, missing) -> Optional[MessageFeedback]:
    msg = get_message(db, message_id)
    if not msg or msg.role != "assistant":
        return None

    fb = db.query(MessageFeedback).filter(
        MessageFeedback.message_id == message_id).first()
    if fb is None:
        fb = MessageFeedback(
            message_id=message_id,
            vote=None if vote is missing else vote,
            comment=None if comment is missing else comment,
        )
        db.add(fb)
    else:
        if vote is not missing:
            fb.vote = vote
        if comment is not missing:
            fb.comment = comment
        fb.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(fb)
    return fb


def get_feedback(db: Session, message_id: int) -> Optional[MessageFeedback]:
    return db.query(MessageFeedback).filter(MessageFeedback.message_id == message_id).first()
