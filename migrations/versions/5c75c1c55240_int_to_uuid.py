"""int -> uuid for chat_sessions.id, chat_messages.id/session_id, message_feedback.message_id

Revision ID: 5c75c1c55240
Revises: 3ddd61f35f38
Create Date: 2026-07-08 23:00:00.000000

"""
from typing import Sequence, Union
from uuid import uuid4

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '5c75c1c55240'
down_revision: Union[str, Sequence[str], None] = '3ddd61f35f38'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()

    # ------------------------------------------------------------------
    # 1. chat_sessions.id + chat_messages.session_id: int -> uuid
    # ------------------------------------------------------------------
    op.add_column(
        "chat_sessions",
        sa.Column("id_new", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "chat_messages",
        sa.Column("session_id_new", postgresql.UUID(
            as_uuid=True), nullable=True),
    )

    session_ids = conn.execute(
        sa.text("SELECT id FROM chat_sessions")).fetchall()
    for row in session_ids:
        new_id = uuid4()
        conn.execute(
            sa.text("UPDATE chat_sessions SET id_new = :new_id WHERE id = :old_id"),
            {"new_id": new_id, "old_id": row.id},
        )
        conn.execute(
            sa.text(
                "UPDATE chat_messages SET session_id_new = :new_id "
                "WHERE session_id = :old_id"
            ),
            {"new_id": new_id, "old_id": row.id},
        )

    op.drop_constraint(
        "chat_messages_session_id_fkey", "chat_messages", type_="foreignkey"
    )
    op.drop_constraint("chat_sessions_pkey", "chat_sessions", type_="primary")
    op.drop_index(op.f("ix_chat_messages_session_id"),
                  table_name="chat_messages")

    op.drop_column("chat_messages", "session_id")
    op.alter_column(
        "chat_messages", "session_id_new", new_column_name="session_id"
    )

    op.drop_column("chat_sessions", "id")
    op.alter_column("chat_sessions", "id_new", new_column_name="id")

    op.alter_column("chat_sessions", "id", nullable=False)
    op.create_primary_key("chat_sessions_pkey", "chat_sessions", ["id"])

    op.alter_column("chat_messages", "session_id", nullable=False)
    op.create_index(
        op.f("ix_chat_messages_session_id"), "chat_messages", ["session_id"]
    )
    op.create_foreign_key(
        "chat_messages_session_id_fkey",
        "chat_messages", "chat_sessions",
        ["session_id"], ["id"],
        ondelete="CASCADE",
    )

    # ------------------------------------------------------------------
    # 2. chat_messages.id + message_feedback.message_id: int -> uuid
    # ------------------------------------------------------------------
    op.add_column(
        "chat_messages",
        sa.Column("id_new", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "message_feedback",
        sa.Column("message_id_new", postgresql.UUID(
            as_uuid=True), nullable=True),
    )

    message_ids = conn.execute(
        sa.text("SELECT id FROM chat_messages")).fetchall()
    for row in message_ids:
        new_id = uuid4()
        conn.execute(
            sa.text("UPDATE chat_messages SET id_new = :new_id WHERE id = :old_id"),
            {"new_id": new_id, "old_id": row.id},
        )
        conn.execute(
            sa.text(
                "UPDATE message_feedback SET message_id_new = :new_id "
                "WHERE message_id = :old_id"
            ),
            {"new_id": new_id, "old_id": row.id},
        )

    op.drop_constraint(
        "message_feedback_message_id_fkey", "message_feedback", type_="foreignkey"
    )
    op.drop_index(
        op.f("ix_message_feedback_message_id"), table_name="message_feedback"
    )
    op.drop_constraint("chat_messages_pkey", "chat_messages", type_="primary")

    op.drop_column("message_feedback", "message_id")
    op.alter_column(
        "message_feedback", "message_id_new", new_column_name="message_id"
    )

    op.drop_column("chat_messages", "id")
    op.alter_column("chat_messages", "id_new", new_column_name="id")

    op.alter_column("chat_messages", "id", nullable=False)
    op.create_primary_key("chat_messages_pkey", "chat_messages", ["id"])

    op.alter_column("message_feedback", "message_id", nullable=False)
    op.create_index(
        op.f("ix_message_feedback_message_id"),
        "message_feedback", ["message_id"], unique=True,
    )
    op.create_foreign_key(
        "message_feedback_message_id_fkey",
        "message_feedback", "chat_messages",
        ["message_id"], ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Обратная конвертация UUID -> int невозможна без потери исходных
    # числовых id (они безвозвратно заменены на UUID при апгрейде).
    # Если нужен откат - восстанавливайте базу из бэкапа, снятого перед
    # апгрейдом.
    raise NotImplementedError(
        "Downgrade не поддерживается для перехода int -> uuid. "
        "Восстанавливайте БД из бэкапа."
    )
