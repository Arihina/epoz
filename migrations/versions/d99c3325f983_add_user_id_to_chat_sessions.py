"""add user_id to chat_sessions

Revision ID: d99c3325f983
Revises: 0223aebe22dc
Create Date: 2026-06-02 13:49:55.053803

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd99c3325f983'
down_revision: Union[str, Sequence[str], None] = '0223aebe22dc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
