"""Store provider-free titles for judged documents.

Revision ID: 20260825_0002
Revises: 20260822_0001
Create Date: 2026-08-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260825_0002"
down_revision: str | None = "20260822_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "judged_document_titles",
        sa.Column("query_set_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.ForeignKeyConstraint(["query_set_id"], ["query_sets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("query_set_id", "document_id"),
    )


def downgrade() -> None:
    op.drop_table("judged_document_titles")
