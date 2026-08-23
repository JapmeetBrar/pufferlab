"""Create immutable revision and eval lifecycle tables.

Revision ID: 20260822_0001
Revises:
Create Date: 2026-08-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260822_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=255), nullable=False),
        sa.Column("corpus_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_dataset_versions_slug_version", "dataset_versions", ["slug", "version"], unique=False
    )

    op.create_table(
        "retrieval_configs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("config_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"], ["dataset_versions.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_retrieval_configs_dataset_name_revision",
        "retrieval_configs",
        ["dataset_version_id", "name", "revision"],
        unique=False,
    )

    op.create_table(
        "query_sets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=255), nullable=False),
        sa.Column("content_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"], ["dataset_versions.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_query_sets_dataset_name_version",
        "query_sets",
        ["dataset_version_id", "name", "version"],
        unique=False,
    )

    op.create_table(
        "judged_queries",
        sa.Column("query_set_id", sa.String(length=36), nullable=False),
        sa.Column("query_id", sa.String(length=36), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["query_set_id"], ["query_sets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("query_set_id", "query_id"),
    )
    op.create_index(
        "ix_judged_queries_set_ordinal",
        "judged_queries",
        ["query_set_id", "ordinal"],
        unique=True,
    )

    op.create_table(
        "qrels",
        sa.Column("query_set_id", sa.String(length=36), nullable=False),
        sa.Column("query_id", sa.String(length=36), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("relevance_grade", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["query_set_id", "query_id"],
            ["judged_queries.query_set_id", "judged_queries.query_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("query_set_id", "query_id", "ordinal"),
    )
    op.create_index(
        "ix_qrels_query_document",
        "qrels",
        ["query_set_id", "query_id", "document_id"],
        unique=False,
    )

    op.create_table(
        "eval_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("query_set_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("completed_queries", sa.Integer(), nullable=False),
        sa.Column("total_queries", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.String(length=32), nullable=True),
        sa.Column("completed_at", sa.String(length=32), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["query_set_id"], ["query_sets.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_eval_runs_status_created", "eval_runs", ["status", "created_at"], unique=False
    )

    op.create_table(
        "run_configs",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("config_id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["config_id"], ["retrieval_configs.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["run_id"], ["eval_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "config_id"),
    )
    op.create_index("ix_run_configs_run_ordinal", "run_configs", ["run_id", "ordinal"], unique=True)

    op.create_table(
        "query_outcomes",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("config_id", sa.String(length=36), nullable=False),
        sa.Column("query_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id", "config_id"],
            ["run_configs.run_id", "run_configs.config_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "config_id", "query_id"),
    )
    op.create_index(
        "ix_query_outcomes_run_query",
        "query_outcomes",
        ["run_id", "query_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_query_outcomes_run_query", table_name="query_outcomes")
    op.drop_table("query_outcomes")
    op.drop_index("ix_run_configs_run_ordinal", table_name="run_configs")
    op.drop_table("run_configs")
    op.drop_index("ix_eval_runs_status_created", table_name="eval_runs")
    op.drop_table("eval_runs")
    op.drop_index("ix_qrels_query_document", table_name="qrels")
    op.drop_table("qrels")
    op.drop_index("ix_judged_queries_set_ordinal", table_name="judged_queries")
    op.drop_table("judged_queries")
    op.drop_index("ix_query_sets_dataset_name_version", table_name="query_sets")
    op.drop_table("query_sets")
    op.drop_index("ix_retrieval_configs_dataset_name_revision", table_name="retrieval_configs")
    op.drop_table("retrieval_configs")
    op.drop_index("ix_dataset_versions_slug_version", table_name="dataset_versions")
    op.drop_table("dataset_versions")
