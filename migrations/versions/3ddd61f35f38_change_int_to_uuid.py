"""change int to uuid

Revision ID: 3ddd61f35f38
Revises: 532f4370e92c
Create Date: 2026-07-08 22:27:50.404305

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3ddd61f35f38'
down_revision: Union[str, Sequence[str], None] = '532f4370e92c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
