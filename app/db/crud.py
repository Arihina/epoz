from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import ChatMessage, MessageFeedback, Conversation


async def create_conversation(db: AsyncSession, user_id, title: Optional[str] = None) -> Conversation:
    c = Conversation(user_id=user_id, title=title)
    db.add(c)
    
    await db.commit()
    await db.refresh(c)

    return c


async def get_conversation(db: AsyncSession, conversation_id: UUID, user_id) -> Optional[Conversation]:
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )

    return result.scalar_one_or_none()


async def list_conversations(db: AsyncSession, user_id) -> list[Conversation]:
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )

    return result.scalars().all()


async def rename_conversation(db: AsyncSession, c: Conversation, title: str) -> Conversation:
    c.title = title

    await db.commit()
    await db.refresh(c)

    return c


async def delete_conversation(db: AsyncSession, c: Conversation) -> None:
    await db.delete(c)
    await db.commit()


async def get_conversation_messages(db: AsyncSession, conversation_id: UUID) -> list[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at)
    )

    return result.scalars().all()


async def touch_conversation(db: AsyncSession, conversation_id: UUID, title: Optional[str] = None) -> None:
    c = await db.get(Conversation, conversation_id)
    if c is None:
        return
    c.updated_at = datetime.now(timezone.utc)
    if title is not None and c.title is None:
        c.title = title
    await db.commit()


async def add_message(
    db: AsyncSession,
    user_id,
    role: str,
    content: str,
    sources: Optional[list[str]] = None,
    retrieved_chunks: Optional[list[dict]] = None,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    id: Optional[UUID] = None,
    conversation_id: Optional[UUID] = None,
) -> ChatMessage:
    msg = ChatMessage(
        id=id or uuid4(),
        user_id=user_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=sources,
        retrieved_chunks=retrieved_chunks,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    db.add(msg)

    await db.commit()
    await db.refresh(msg)

    return msg


async def get_message(db: AsyncSession, message_id: UUID) -> Optional[ChatMessage]:
    result = await db.execute(select(ChatMessage).where(ChatMessage.id == message_id))
    return result.scalar_one_or_none()


async def get_message_for_user(db: AsyncSession, message_id: UUID, user_id) -> Optional[ChatMessage]:
    result = await db.execute(
        select(ChatMessage).where(
            ChatMessage.id == message_id,
            ChatMessage.user_id == user_id,
        )
    )

    return result.scalar_one_or_none()


async def upsert_feedback(
    db: AsyncSession,
    message_id: UUID,
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


async def get_feedback(db: AsyncSession, message_id: UUID) -> Optional[MessageFeedback]:
    result = await db.execute(
        select(MessageFeedback).where(MessageFeedback.message_id == message_id)
    )
    return result.scalar_one_or_none()
