"""add status to generated documents

Revision ID: d4e5f6a7b8c9
Revises: c1d2e3f4a5b6
Create Date: 2026-11-16 10:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, Sequence[str], None] = 'c1d2e3f4a5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('generated_documents', sa.Column('instructions', sa.Text(), nullable=True))
    op.add_column('generated_documents', sa.Column('status', sa.String(length=16), nullable=False, server_default='pending'))
    op.add_column('generated_documents', sa.Column('error_message', sa.Text(), nullable=True))
    op.create_check_constraint(
        'ck_generated_documents_status', 'generated_documents',
        "status IN ('pending', 'processing', 'done', 'failed')")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('ck_generated_documents_status', 'generated_documents', type_='check')
    op.drop_column('generated_documents', 'error_message')
    op.drop_column('generated_documents', 'status')
    op.drop_column('generated_documents', 'instructions')
