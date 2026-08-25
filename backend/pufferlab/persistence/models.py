"""SQLAlchemy rows for local control-plane and evaluation state."""

from sqlalchemy import ForeignKey, ForeignKeyConstraint, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DatasetVersionRow(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(255), nullable=False)
    corpus_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (Index("ix_dataset_versions_slug_version", "slug", "version"),)


class RetrievalConfigRow(Base):
    __tablename__ = "retrieval_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("dataset_versions.id", ondelete="RESTRICT"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        Index(
            "ix_retrieval_configs_dataset_name_revision", "dataset_version_id", "name", "revision"
        ),
    )


class QuerySetRow(Base):
    __tablename__ = "query_sets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("dataset_versions.id", ondelete="RESTRICT"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        Index("ix_query_sets_dataset_name_version", "dataset_version_id", "name", "version"),
    )


class JudgedQueryRow(Base):
    __tablename__ = "judged_queries"

    query_set_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("query_sets.id", ondelete="CASCADE"), primary_key=True
    )
    query_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        Index("ix_judged_queries_set_ordinal", "query_set_id", "ordinal", unique=True),
    )


class QrelRow(Base):
    __tablename__ = "qrels"

    query_set_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[str] = mapped_column(String(36), nullable=False)
    relevance_grade: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["query_set_id", "query_id"],
            ["judged_queries.query_set_id", "judged_queries.query_id"],
            ondelete="CASCADE",
        ),
        Index("ix_qrels_query_document", "query_set_id", "query_id", "document_id"),
    )


class JudgedDocumentTitleRow(Base):
    __tablename__ = "judged_document_titles"

    query_set_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("query_sets.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)


class EvalRunRow(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query_set_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("query_sets.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    completed_queries: Mapped[int] = mapped_column(Integer, nullable=False)
    total_queries: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    completed_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (Index("ix_eval_runs_status_created", "status", "created_at"),)


class RunConfigRow(Base):
    __tablename__ = "run_configs"

    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("eval_runs.id", ondelete="CASCADE"), primary_key=True
    )
    config_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("retrieval_configs.id", ondelete="RESTRICT"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (Index("ix_run_configs_run_ordinal", "run_id", "ordinal", unique=True),)


class QueryOutcomeRow(Base):
    __tablename__ = "query_outcomes"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    config_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "config_id"],
            ["run_configs.run_id", "run_configs.config_id"],
            ondelete="CASCADE",
        ),
        Index("ix_query_outcomes_run_query", "run_id", "query_id"),
    )
