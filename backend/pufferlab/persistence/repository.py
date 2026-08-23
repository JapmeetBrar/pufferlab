"""Transaction-scoped repositories for immutable revisions and eval lifecycle state."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, sessionmaker

from pufferlab.contracts.datasets import DatasetVersion
from pufferlab.contracts.errors import ApiErrorDetail
from pufferlab.contracts.evals import (
    ConfigRunSummary,
    EvalRun,
    EvalRunStatus,
    JudgedQuery,
    Qrel,
    QuerySet,
    QuerySetSummary,
)
from pufferlab.contracts.retrieval import RetrievalConfig
from pufferlab.persistence.canonical import canonical_json, canonical_utc
from pufferlab.persistence.errors import (
    ImmutableRecordError,
    InvalidRunTransitionError,
    PersistenceValidationError,
    RecordNotFoundError,
)
from pufferlab.persistence.models import (
    DatasetVersionRow,
    EvalRunRow,
    JudgedQueryRow,
    QrelRow,
    QueryOutcomeRow,
    QuerySetRow,
    RetrievalConfigRow,
    RunConfigRow,
)
from pufferlab.persistence.types import QueryOutcome

_TERMINAL_STATUSES = {
    EvalRunStatus.COMPLETED,
    EvalRunStatus.FAILED,
    EvalRunStatus.CANCELLED,
    EvalRunStatus.INTERRUPTED,
}

_ALLOWED_TRANSITIONS = {
    EvalRunStatus.QUEUED: {
        EvalRunStatus.RUNNING,
        EvalRunStatus.CANCELLED,
        EvalRunStatus.INTERRUPTED,
    },
    EvalRunStatus.RUNNING: {
        EvalRunStatus.COMPLETED,
        EvalRunStatus.FAILED,
        EvalRunStatus.CANCELLED,
        EvalRunStatus.INTERRUPTED,
    },
}


class PufferLabRepository:
    """Persist one method call in one explicit SQLite transaction."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def put_dataset_version(self, value: DatasetVersion) -> DatasetVersion:
        payload = canonical_json(value)
        created_at = canonical_utc(value.created_at, field_name="dataset_version.created_at")
        with self._session_factory.begin() as session:
            existing = session.get(DatasetVersionRow, str(value.id))
            if existing is not None:
                self._require_same_payload(
                    existing.payload_json, payload, "dataset version", value.id
                )
                return DatasetVersion.model_validate_json(existing.payload_json)
            session.add(
                DatasetVersionRow(
                    id=str(value.id),
                    slug=value.slug,
                    version=value.version,
                    corpus_hash=value.corpus_hash,
                    created_at=created_at,
                    payload_json=payload,
                )
            )
        return value

    def get_dataset_version(self, dataset_version_id: UUID) -> DatasetVersion:
        with self._session_factory() as session:
            row = session.get(DatasetVersionRow, str(dataset_version_id))
            if row is None:
                raise RecordNotFoundError(f"dataset version {dataset_version_id} was not found")
            return DatasetVersion.model_validate_json(row.payload_json)

    def put_retrieval_config(self, value: RetrievalConfig) -> RetrievalConfig:
        payload = canonical_json(value)
        created_at = canonical_utc(value.created_at, field_name="retrieval_config.created_at")
        with self._session_factory.begin() as session:
            existing = session.get(RetrievalConfigRow, str(value.id))
            if existing is not None:
                self._require_same_payload(
                    existing.payload_json, payload, "retrieval config", value.id
                )
                return RetrievalConfig.model_validate_json(existing.payload_json)
            self._require_dataset(session, value.dataset_version_id)
            session.add(
                RetrievalConfigRow(
                    id=str(value.id),
                    revision=value.revision,
                    dataset_version_id=str(value.dataset_version_id),
                    name=value.name,
                    config_hash=value.config_hash,
                    created_at=created_at,
                    payload_json=payload,
                )
            )
        return value

    def get_retrieval_config(self, config_id: UUID) -> RetrievalConfig:
        with self._session_factory() as session:
            row = session.get(RetrievalConfigRow, str(config_id))
            if row is None:
                raise RecordNotFoundError(f"retrieval config {config_id} was not found")
            return RetrievalConfig.model_validate_json(row.payload_json)

    def put_query_set(
        self,
        value: QuerySet,
        judged_queries: Sequence[JudgedQuery],
    ) -> tuple[QuerySet, list[JudgedQuery]]:
        if value.query_count != len(judged_queries):
            raise PersistenceValidationError(
                "query_set.query_count must equal the number of judged queries"
            )
        query_ids = [query.id for query in judged_queries]
        if len(set(query_ids)) != len(query_ids):
            raise PersistenceValidationError("judged query IDs must be unique within a query set")

        payload = canonical_json(value)
        created_at = canonical_utc(value.created_at, field_name="query_set.created_at")
        with self._session_factory.begin() as session:
            existing = session.get(QuerySetRow, str(value.id))
            if existing is not None:
                persisted = self._load_query_set(session, existing)
                if canonical_json(persisted[0]) != payload or [
                    canonical_json(query) for query in persisted[1]
                ] != [canonical_json(query) for query in judged_queries]:
                    raise ImmutableRecordError(f"query set {value.id} is an immutable revision")
                return persisted

            self._require_dataset(session, value.dataset_version_id)
            session.add(
                QuerySetRow(
                    id=str(value.id),
                    dataset_version_id=str(value.dataset_version_id),
                    name=value.name,
                    version=value.version,
                    content_hash=value.content_hash,
                    created_at=created_at,
                    payload_json=payload,
                )
            )
            session.flush()
            for query_ordinal, query in enumerate(judged_queries):
                query_payload = canonical_json(query.model_dump(mode="json", exclude={"qrels"}))
                session.add(
                    JudgedQueryRow(
                        query_set_id=str(value.id),
                        query_id=str(query.id),
                        ordinal=query_ordinal,
                        payload_json=query_payload,
                    )
                )
                session.flush()
                for qrel_ordinal, qrel in enumerate(query.qrels):
                    session.add(
                        QrelRow(
                            query_set_id=str(value.id),
                            query_id=str(query.id),
                            ordinal=qrel_ordinal,
                            document_id=str(qrel.document_id),
                            relevance_grade=qrel.relevance_grade,
                        )
                    )
        return value, list(judged_queries)

    def get_query_set(self, query_set_id: UUID) -> tuple[QuerySet, list[JudgedQuery]]:
        with self._session_factory() as session:
            row = session.get(QuerySetRow, str(query_set_id))
            if row is None:
                raise RecordNotFoundError(f"query set {query_set_id} was not found")
            return self._load_query_set(session, row)

    def create_run(self, value: EvalRun) -> EvalRun:
        self._validate_new_run_shape(value)
        payload = canonical_json(value)
        with self._session_factory.begin() as session:
            existing = session.get(EvalRunRow, str(value.id))
            if existing is not None:
                self._require_same_payload(existing.payload_json, payload, "eval run", value.id)
                return EvalRun.model_validate_json(existing.payload_json)

            query_set_row = session.get(QuerySetRow, str(value.query_set.id))
            if query_set_row is None:
                raise PersistenceValidationError(
                    f"query set {value.query_set.id} must be persisted before its run"
                )
            query_set = QuerySet.model_validate_json(query_set_row.payload_json)
            expected_summary = QuerySetSummary(
                id=query_set.id,
                name=query_set.name,
                version=query_set.version,
                query_count=query_set.query_count,
                content_hash=query_set.content_hash,
            )
            if value.query_set != expected_summary:
                raise PersistenceValidationError(
                    "run query-set summary does not match its revision"
                )
            if value.total_queries != query_set.query_count:
                raise PersistenceValidationError(
                    "run total_queries must match the query-set revision"
                )

            config_ids = [value.baseline_config_id, *value.candidate_config_ids]
            if len(set(config_ids)) != len(config_ids):
                raise PersistenceValidationError("run config IDs must be unique")
            configs = [self._require_config(session, config_id) for config_id in config_ids]
            if any(config.dataset_version_id != query_set.dataset_version_id for config in configs):
                raise PersistenceValidationError(
                    "every run config must reference the query set's dataset version"
                )

            session.add(
                EvalRunRow(
                    id=str(value.id),
                    query_set_id=str(value.query_set.id),
                    status=value.status.value,
                    completed_queries=value.completed_queries,
                    total_queries=value.total_queries,
                    created_at=canonical_utc(value.created_at, field_name="eval_run.created_at"),
                    started_at=None,
                    completed_at=None,
                    payload_json=payload,
                )
            )
            session.flush()
            for ordinal, config_id in enumerate(config_ids):
                session.add(
                    RunConfigRow(
                        run_id=str(value.id),
                        config_id=str(config_id),
                        role="baseline" if ordinal == 0 else "candidate",
                        ordinal=ordinal,
                    )
                )
        return value

    def get_run(self, run_id: UUID) -> EvalRun:
        with self._session_factory() as session:
            row = self._require_run(session, run_id)
            return EvalRun.model_validate_json(row.payload_json)

    def transition_run(
        self,
        run_id: UUID,
        target: EvalRunStatus,
        *,
        at: datetime | None = None,
        summaries: Sequence[ConfigRunSummary] | None = None,
        error: ApiErrorDetail | None = None,
    ) -> EvalRun:
        transition_at = at or datetime.now(UTC)
        canonical_utc(transition_at, field_name="run_transition.at")
        with self._session_factory.begin() as session:
            row = self._require_run(session, run_id)
            return self._transition_row(
                row,
                target,
                transition_at=transition_at,
                summaries=summaries,
                error=error,
            )

    def record_outcome(self, outcome: QueryOutcome) -> EvalRun:
        payload = canonical_json(outcome)
        created_at = canonical_utc(outcome.created_at, field_name="query_outcome.created_at")
        with self._session_factory.begin() as session:
            run_row = self._require_run(session, outcome.run_id)
            run = EvalRun.model_validate_json(run_row.payload_json)
            if run.status is not EvalRunStatus.RUNNING:
                raise ImmutableRecordError(
                    f"run {run.id} is {run.status.value}; outcomes require a running run"
                )
            if session.get(RunConfigRow, (str(run.id), str(outcome.config_id))) is None:
                raise PersistenceValidationError("outcome config is not part of the run")
            query_key = (str(run.query_set.id), str(outcome.query_id))
            if session.get(JudgedQueryRow, query_key) is None:
                raise PersistenceValidationError("outcome query is not part of the run's query set")

            key = (str(outcome.run_id), str(outcome.config_id), str(outcome.query_id))
            outcome_row = session.get(QueryOutcomeRow, key)
            if outcome_row is None:
                session.add(
                    QueryOutcomeRow(
                        run_id=key[0],
                        config_id=key[1],
                        query_id=key[2],
                        status=outcome.status.value,
                        created_at=created_at,
                        payload_json=payload,
                    )
                )
            else:
                outcome_row.status = outcome.status.value
                outcome_row.created_at = created_at
                outcome_row.payload_json = payload

            session.flush()
            completed_queries = self._completed_query_count(session, run.id)
            updated = run.model_copy(update={"completed_queries": completed_queries})
            run_row.completed_queries = completed_queries
            run_row.payload_json = canonical_json(updated)
        return updated

    def get_outcome(self, run_id: UUID, config_id: UUID, query_id: UUID) -> QueryOutcome:
        with self._session_factory() as session:
            row = session.get(
                QueryOutcomeRow,
                (str(run_id), str(config_id), str(query_id)),
            )
            if row is None:
                raise RecordNotFoundError("query outcome was not found")
            return QueryOutcome.model_validate_json(row.payload_json)

    def list_outcomes(self, run_id: UUID) -> list[QueryOutcome]:
        with self._session_factory() as session:
            self._require_run(session, run_id)
            rows = session.scalars(
                select(QueryOutcomeRow)
                .where(QueryOutcomeRow.run_id == str(run_id))
                .order_by(QueryOutcomeRow.query_id, QueryOutcomeRow.config_id)
            ).all()
            return [QueryOutcome.model_validate_json(row.payload_json) for row in rows]

    def interrupt_stale_runs(self, *, at: datetime | None = None) -> list[UUID]:
        """Fail closed on process-owned queued/running work after a restart."""
        transition_at = at or datetime.now(UTC)
        canonical_utc(transition_at, field_name="startup_recovery.at")
        interrupted: list[UUID] = []
        with self._session_factory.begin() as session:
            rows = session.scalars(
                select(EvalRunRow)
                .where(
                    EvalRunRow.status.in_([EvalRunStatus.QUEUED.value, EvalRunStatus.RUNNING.value])
                )
                .order_by(EvalRunRow.created_at, EvalRunRow.id)
            ).all()
            for row in rows:
                self._transition_row(
                    row,
                    EvalRunStatus.INTERRUPTED,
                    transition_at=transition_at,
                    summaries=None,
                    error=None,
                )
                interrupted.append(UUID(row.id))
        return interrupted

    @staticmethod
    def _validate_new_run_shape(value: EvalRun) -> None:
        if value.status is not EvalRunStatus.QUEUED:
            raise PersistenceValidationError("new runs must start queued")
        if value.completed_queries != 0:
            raise PersistenceValidationError("new runs must start with zero completed queries")
        if value.started_at is not None or value.completed_at is not None:
            raise PersistenceValidationError("new queued runs cannot have lifecycle timestamps")
        if value.error is not None or value.summaries:
            raise PersistenceValidationError("new queued runs cannot have summaries or an error")
        canonical_utc(value.created_at, field_name="eval_run.created_at")

    @staticmethod
    def _require_same_payload(
        actual: str,
        expected: str,
        record_type: str,
        record_id: UUID,
    ) -> None:
        if actual != expected:
            raise ImmutableRecordError(f"{record_type} {record_id} is an immutable revision")

    @staticmethod
    def _require_dataset(session: Session, dataset_version_id: UUID) -> DatasetVersion:
        row = session.get(DatasetVersionRow, str(dataset_version_id))
        if row is None:
            raise PersistenceValidationError(
                f"dataset version {dataset_version_id} must be persisted first"
            )
        return DatasetVersion.model_validate_json(row.payload_json)

    @staticmethod
    def _require_config(session: Session, config_id: UUID) -> RetrievalConfig:
        row = session.get(RetrievalConfigRow, str(config_id))
        if row is None:
            raise PersistenceValidationError(
                f"retrieval config {config_id} must be persisted first"
            )
        return RetrievalConfig.model_validate_json(row.payload_json)

    @staticmethod
    def _require_run(session: Session, run_id: UUID) -> EvalRunRow:
        row = session.get(EvalRunRow, str(run_id))
        if row is None:
            raise RecordNotFoundError(f"eval run {run_id} was not found")
        return row

    @staticmethod
    def _load_query_set(
        session: Session,
        row: QuerySetRow,
    ) -> tuple[QuerySet, list[JudgedQuery]]:
        value = QuerySet.model_validate_json(row.payload_json)
        query_rows = session.scalars(
            select(JudgedQueryRow)
            .where(JudgedQueryRow.query_set_id == row.id)
            .order_by(JudgedQueryRow.ordinal)
        ).all()
        queries: list[JudgedQuery] = []
        for query_row in query_rows:
            qrels = session.scalars(
                select(QrelRow)
                .where(
                    QrelRow.query_set_id == row.id,
                    QrelRow.query_id == query_row.query_id,
                )
                .order_by(QrelRow.ordinal)
            ).all()
            query_data = json.loads(query_row.payload_json)
            query_data["qrels"] = [
                Qrel(document_id=UUID(qrel.document_id), relevance_grade=qrel.relevance_grade)
                for qrel in qrels
            ]
            queries.append(JudgedQuery.model_validate(query_data))
        return value, queries

    def _transition_row(
        self,
        row: EvalRunRow,
        target: EvalRunStatus,
        *,
        transition_at: datetime,
        summaries: Sequence[ConfigRunSummary] | None,
        error: ApiErrorDetail | None,
    ) -> EvalRun:
        current = EvalRun.model_validate_json(row.payload_json)
        allowed = _ALLOWED_TRANSITIONS.get(current.status, set())
        if target not in allowed:
            raise InvalidRunTransitionError(
                f"run transition {current.status.value} -> {target.value} is not allowed"
            )
        if target is EvalRunStatus.COMPLETED and current.completed_queries != current.total_queries:
            raise InvalidRunTransitionError("a run cannot complete before every query is durable")
        if error is not None and target is not EvalRunStatus.FAILED:
            raise PersistenceValidationError("only failed runs may store a run-level error")

        updates: dict[str, object] = {"status": target}
        if target is EvalRunStatus.RUNNING:
            updates["started_at"] = transition_at
        if target in _TERMINAL_STATUSES:
            updates["completed_at"] = transition_at
        if summaries is not None:
            updates["summaries"] = list(summaries)
        if target is EvalRunStatus.FAILED:
            updates["error"] = error

        updated = current.model_copy(update=updates)
        row.status = target.value
        row.started_at = (
            canonical_utc(updated.started_at, field_name="eval_run.started_at")
            if updated.started_at is not None
            else None
        )
        row.completed_at = (
            canonical_utc(updated.completed_at, field_name="eval_run.completed_at")
            if updated.completed_at is not None
            else None
        )
        row.payload_json = canonical_json(updated)
        return updated

    @staticmethod
    def _completed_query_count(session: Session, run_id: UUID) -> int:
        config_count = session.scalar(
            select(func.count()).select_from(RunConfigRow).where(RunConfigRow.run_id == str(run_id))
        )
        if not config_count:
            return 0
        completed_query_ids: Select[tuple[str]] = (
            select(QueryOutcomeRow.query_id)
            .where(QueryOutcomeRow.run_id == str(run_id))
            .group_by(QueryOutcomeRow.query_id)
            .having(func.count(QueryOutcomeRow.config_id) == config_count)
        )
        return int(
            session.scalar(select(func.count()).select_from(completed_query_ids.subquery())) or 0
        )
