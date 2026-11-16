"""add files

Revision ID: 7b3f9c1a2d4e
Revises: eae6c1ed4878
Create Date: 2026-11-16 08:40:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7b3f9c1a2d4e'
down_revision: Union[str, Sequence[str], None] = 'eae6c1ed4878'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'files',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('conversation_id', sa.UUID(), nullable=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('mime_type', sa.String(length=127), nullable=True),
        sa.Column('size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('content', sa.LargeBinary(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('markdown_content', sa.Text(), nullable=True),
        sa.Column('markdown_tokens', sa.Integer(), nullable=True),
        sa.Column('ocr_backend', sa.String(length=64), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'done', 'failed')",
            name='ck_file_status'),
        sa.ForeignKeyConstraint(
            ['conversation_id'], ['conversations.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_files_user_id'), 'files', ['user_id'], unique=False)
    op.create_index(op.f('ix_files_conversation_id'), 'files', ['conversation_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_files_conversation_id'), table_name='files')
    op.drop_index(op.f('ix_files_user_id'), table_name='files')
    op.drop_table('files')
