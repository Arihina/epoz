import json
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import ChatSession, ChatMessage, MessageFeedback


async def create_session(db: AsyncSession, user_id, title: Optional[str] = None) -> ChatSession:
    s = ChatSession(user_id=user_id, title=title)
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return s


async def get_session(db: AsyncSession, session_id: int, user_id) -> Optional[ChatSession]:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def list_sessions(db: AsyncSession, user_id) -> list[ChatSession]:
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
    )
    return result.scalars().all()


async def rename_session(db: AsyncSession, s: ChatSession, title: str) -> ChatSession:
    s.title = title
    await db.commit()
    await db.refresh(s)
    return s


async def delete_session(db: AsyncSession, s: ChatSession) -> None:
    await db.delete(s)
    await db.commit()


async def get_message_for_user(db: AsyncSession, message_id: int, user_id) -> Optional[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .join(ChatSession, ChatMessage.session_id == ChatSession.id)
        .where(
            ChatMessage.id == message_id,
            ChatSession.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def _touch(db: AsyncSession, session_id: int) -> None:
    await db.execute(
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(updated_at=datetime.now(timezone.utc))
    )
    await db.commit()


async def add_message(
    db: AsyncSession,
    session_id: int,
    role: str,
    content: str,
    sources: Optional[list[str]] = None,
) -> ChatMessage:
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sources=sources,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    await _touch(db, session_id)
    return msg


async def get_messages(db: AsyncSession, session_id: int) -> list[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
    )
    return result.scalars().all()


async def get_message(db: AsyncSession, message_id: int) -> Optional[ChatMessage]:
    result = await db.execute(select(ChatMessage).where(ChatMessage.id == message_id))
    return result.scalar_one_or_none()


async def build_history(db: AsyncSession, session_id: int) -> list[tuple[str, str]]:
    messages = await get_messages(db, session_id)
    return [(m.role, m.content) for m in messages]


async def upsert_feedback(
    db: AsyncSession,
    message_id: int,
    vote,
    comment,
    missing,
) -> Optional[MessageFeedback]:
    msg = await get_message(db, message_id)
    if not msg or msg.role != "assistant":
        return None

    result = await db.execute(
        select(MessageFeedback).where(MessageFeedback.message_id == message_id)
    )
    fb = result.scalar_one_or_none()

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

    await db.commit()
    await db.refresh(fb)
    return fb


async def get_feedback(db: AsyncSession, message_id: int) -> Optional[MessageFeedback]:
    result = await db.execute(
        select(MessageFeedback).where(MessageFeedback.message_id == message_id)
    )
    return result.scalar_one_or_none()
