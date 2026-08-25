"""Transaction-scoped repositories for immutable revisions and eval lifecycle state."""

import json
from collections.abc import Mapping, Sequence
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
    JudgedDocumentTitleRow,
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

_MAX_CATALOG_ROWS = 100
_MAX_RUN_ROWS = 100
_MAX_OUTCOME_ROWS = 200
_MAX_QUERY_SET_JUDGED_DOCUMENT_TITLES = 5_000
_MAX_QUERY_JUDGED_DOCUMENT_TITLES = 100

_ALLOWED_TRANSITIONS = {
    EvalRunStatus.QUEUED: {
        EvalRunStatus.RUNNING,
        EvalRunStatus.FAILED,
        EvalRunStatus.CANCELLED,
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
                return self._decode_dataset_row(existing)
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
            return self._decode_dataset_row(row)

    def list_dataset_versions(self, *, limit: int = _MAX_CATALOG_ROWS) -> list[DatasetVersion]:
        """Return immutable dataset revisions in a deterministic selection order."""
        self._validate_limit(limit, maximum=_MAX_CATALOG_ROWS)
        with self._session_factory() as session:
            rows = session.scalars(
                select(DatasetVersionRow)
                .order_by(
                    DatasetVersionRow.created_at,
                    DatasetVersionRow.id,
                )
                .limit(limit)
            ).all()
            return [self._decode_dataset_row(row) for row in rows]

    def put_retrieval_config(self, value: RetrievalConfig) -> RetrievalConfig:
        payload = canonical_json(value)
        created_at = canonical_utc(value.created_at, field_name="retrieval_config.created_at")
        with self._session_factory.begin() as session:
            existing = session.get(RetrievalConfigRow, str(value.id))
            if existing is not None:
                self._require_same_payload(
                    existing.payload_json, payload, "retrieval config", value.id
                )
                return self._decode_config_row(existing)
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
            return self._decode_config_row(row)

    def list_retrieval_configs(
        self,
        *,
        dataset_version_id: UUID | None = None,
        limit: int = _MAX_CATALOG_ROWS,
    ) -> list[RetrievalConfig]:
        """Return immutable config revisions, optionally scoped to one dataset revision."""
        self._validate_limit(limit, maximum=_MAX_CATALOG_ROWS)
        with self._session_factory() as session:
            statement = select(RetrievalConfigRow)
            if dataset_version_id is not None:
                statement = statement.where(
                    RetrievalConfigRow.dataset_version_id == str(dataset_version_id)
                )
            rows = session.scalars(
                statement.order_by(
                    RetrievalConfigRow.created_at,
                    RetrievalConfigRow.id,
                ).limit(limit)
            ).all()
            return [self._decode_config_row(row) for row in rows]

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

    def put_judged_document_titles(
        self,
        query_set_id: UUID,
        titles: Mapping[UUID, str],
    ) -> dict[UUID, str]:
        """Idempotently attach immutable display titles to documents judged in one query set."""
        if len(titles) > _MAX_QUERY_SET_JUDGED_DOCUMENT_TITLES:
            raise PersistenceValidationError("judged-document title count exceeds its bound")
        normalized = {
            document_id: self._validate_document_title(title)
            for document_id, title in titles.items()
        }
        with self._session_factory.begin() as session:
            if session.get(QuerySetRow, str(query_set_id)) is None:
                raise RecordNotFoundError(f"query set {query_set_id} was not found")
            judged_ids = {
                UUID(value)
                for value in session.scalars(
                    select(QrelRow.document_id).where(QrelRow.query_set_id == str(query_set_id))
                ).all()
            }
            if not set(normalized).issubset(judged_ids):
                raise PersistenceValidationError(
                    "judged-document titles must reference qrels in the query set"
                )
            for document_id, title in normalized.items():
                identity = (str(query_set_id), str(document_id))
                existing = session.get(JudgedDocumentTitleRow, identity)
                if existing is not None:
                    if existing.title != title:
                        raise ImmutableRecordError(
                            f"judged-document title {document_id} is immutable"
                        )
                    continue
                session.add(
                    JudgedDocumentTitleRow(
                        query_set_id=str(query_set_id),
                        document_id=str(document_id),
                        title=title,
                    )
                )
        return normalized

    def get_judged_document_titles(
        self,
        query_set_id: UUID,
        document_ids: Sequence[UUID],
    ) -> dict[UUID, str]:
        """Load bounded provider-free title snapshots for the requested judged documents."""
        if len(document_ids) > _MAX_QUERY_JUDGED_DOCUMENT_TITLES:
            raise PersistenceValidationError("judged-document title count exceeds its bound")
        requested = set(document_ids)
        with self._session_factory() as session:
            if session.get(QuerySetRow, str(query_set_id)) is None:
                raise RecordNotFoundError(f"query set {query_set_id} was not found")
            if not requested:
                return {}
            rows = session.scalars(
                select(JudgedDocumentTitleRow).where(
                    JudgedDocumentTitleRow.query_set_id == str(query_set_id),
                    JudgedDocumentTitleRow.document_id.in_(
                        tuple(str(value) for value in requested)
                    ),
                )
            ).all()
            try:
                return {
                    UUID(row.document_id): self._validate_document_title(row.title) for row in rows
                }
            except ValueError:
                raise PersistenceValidationError(
                    "stored judged-document title payload is invalid"
                ) from None

    def get_query_set_revision(self, query_set_id: UUID) -> QuerySet:
        """Return query-set metadata without loading licensed query text or qrels."""
        with self._session_factory() as session:
            row = session.get(QuerySetRow, str(query_set_id))
            if row is None:
                raise RecordNotFoundError(f"query set {query_set_id} was not found")
            return self._decode_query_set_row(row)

    def get_judged_query(self, query_set_id: UUID, query_id: UUID) -> JudgedQuery:
        """Return one exact judged query only within its immutable query-set scope."""
        with self._session_factory() as session:
            query_set_row = session.get(QuerySetRow, str(query_set_id))
            if query_set_row is None:
                raise RecordNotFoundError(f"query set {query_set_id} was not found")
            self._decode_query_set_row(query_set_row)
            row = session.get(JudgedQueryRow, (str(query_set_id), str(query_id)))
            if row is None:
                raise RecordNotFoundError("judged query was not found in the requested query set")
            return self._load_judged_query(session, row)

    def list_query_ids(
        self,
        query_set_id: UUID,
        *,
        limit: int = _MAX_CATALOG_ROWS,
    ) -> list[UUID]:
        """Return only bounded indexed identities for one immutable query-set revision."""
        self._validate_limit(limit, maximum=_MAX_CATALOG_ROWS)
        with self._session_factory() as session:
            query_set_row = session.get(QuerySetRow, str(query_set_id))
            if query_set_row is None:
                raise RecordNotFoundError(f"query set {query_set_id} was not found")
            query_set = self._decode_query_set_row(query_set_row)
            rows = session.execute(
                select(JudgedQueryRow.query_id, JudgedQueryRow.ordinal)
                .where(JudgedQueryRow.query_set_id == str(query_set_id))
                .order_by(JudgedQueryRow.ordinal)
                .limit(limit + 1)
            ).all()
            if len(rows) > limit:
                raise PersistenceValidationError(
                    "judged-query identity selection exceeds its bound"
                )
            try:
                query_ids = [UUID(query_id) for query_id, _ordinal in rows]
            except ValueError:
                raise PersistenceValidationError(
                    "stored judged-query identity is invalid"
                ) from None
            ordinals = [ordinal for _query_id, ordinal in rows]
            if (
                len(query_ids) != query_set.query_count
                or len(query_ids) != len(set(query_ids))
                or ordinals != list(range(len(rows)))
            ):
                raise PersistenceValidationError(
                    "stored judged-query identities do not match the query-set revision"
                )
            return query_ids

    def list_query_sets(
        self,
        *,
        dataset_version_id: UUID | None = None,
        limit: int = _MAX_CATALOG_ROWS,
    ) -> list[QuerySet]:
        """Return query-set revisions without loading licensed query or qrel payloads."""
        self._validate_limit(limit, maximum=_MAX_CATALOG_ROWS)
        with self._session_factory() as session:
            statement = select(QuerySetRow)
            if dataset_version_id is not None:
                statement = statement.where(
                    QuerySetRow.dataset_version_id == str(dataset_version_id)
                )
            rows = session.scalars(
                statement.order_by(QuerySetRow.created_at, QuerySetRow.id).limit(limit)
            ).all()
            return [self._decode_query_set_row(row) for row in rows]

    def create_run(self, value: EvalRun) -> EvalRun:
        self._validate_new_run_shape(value)
        payload = canonical_json(value)
        with self._session_factory.begin() as session:
            existing = session.get(EvalRunRow, str(value.id))
            if existing is not None:
                self._require_same_payload(existing.payload_json, payload, "eval run", value.id)
                return self._decode_run_row(existing)

            query_set_row = session.get(QuerySetRow, str(value.query_set.id))
            if query_set_row is None:
                raise PersistenceValidationError(
                    f"query set {value.query_set.id} must be persisted before its run"
                )
            query_set = self._decode_query_set_row(query_set_row)
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
            return self._decode_run_row(row)

    def list_runs(self, *, limit: int = 50) -> list[EvalRun]:
        """Return bounded run revisions newest first with UUID tie-breaking."""
        self._validate_limit(limit, maximum=_MAX_RUN_ROWS)
        with self._session_factory() as session:
            rows = session.scalars(
                select(EvalRunRow)
                .order_by(EvalRunRow.created_at.desc(), EvalRunRow.id)
                .limit(limit)
            ).all()
            return [self._decode_run_row(row) for row in rows]

    def list_active_runs(self, *, limit: int = _MAX_RUN_ROWS) -> list[EvalRun]:
        """Return every bounded queued/running revision in deterministic claim order."""
        self._validate_limit(limit, maximum=_MAX_RUN_ROWS)
        with self._session_factory() as session:
            rows = session.scalars(
                select(EvalRunRow)
                .where(
                    EvalRunRow.status.in_((EvalRunStatus.QUEUED.value, EvalRunStatus.RUNNING.value))
                )
                .order_by(EvalRunRow.created_at, EvalRunRow.id)
                .limit(limit + 1)
            ).all()
            active = [self._decode_run_row(row) for row in rows]
            if len(active) > limit:
                raise PersistenceValidationError("active run selection exceeds its bound")
            return active

    def list_run_configs(self, run_id: UUID) -> list[RetrievalConfig]:
        """Return the immutable run catalog in persisted baseline/candidate order."""
        with self._session_factory() as session:
            run_row = self._require_run(session, run_id)
            run = self._decode_run_row(run_row)
            rows = session.execute(
                select(RunConfigRow, RetrievalConfigRow)
                .join(RetrievalConfigRow, RetrievalConfigRow.id == RunConfigRow.config_id)
                .where(RunConfigRow.run_id == str(run_id))
                .order_by(RunConfigRow.ordinal)
            ).all()
            expected_ids = [run.baseline_config_id, *run.candidate_config_ids]
            try:
                actual_ids = [UUID(run_config.config_id) for run_config, _config in rows]
            except ValueError:
                raise PersistenceValidationError(
                    "stored run-config binding identity is invalid"
                ) from None
            roles = [run_config.role for run_config, _config in rows]
            ordinals = [run_config.ordinal for run_config, _config in rows]
            if (
                actual_ids != expected_ids
                or roles != ["baseline", *("candidate" for _ in run.candidate_config_ids)]
                or ordinals != list(range(len(rows)))
            ):
                raise PersistenceValidationError(
                    "stored run-config bindings do not match the durable run payload"
                )
            return [self._decode_config_row(config) for _run_config, config in rows]

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
            if target is EvalRunStatus.COMPLETED:
                self._validate_completion_summaries(session, row, summaries)
            return self._transition_row(
                row,
                target,
                transition_at=transition_at,
                summaries=summaries,
                error=error,
            )

    def claim_queued_run(
        self,
        run_id: UUID,
        *,
        at: datetime | None = None,
    ) -> EvalRun:
        """Atomically claim exactly one queued revision for the local API worker."""
        transition_at = at or datetime.now(UTC)
        canonical_utc(transition_at, field_name="run_claim.at")
        with self._session_factory.begin() as session:
            row = self._require_run(session, run_id)
            current = self._decode_run_row(row)
            if current.status is not EvalRunStatus.QUEUED:
                raise InvalidRunTransitionError(
                    f"run {run_id} is {current.status.value}; only queued runs may be claimed"
                )
            return self._transition_row(
                row,
                EvalRunStatus.RUNNING,
                transition_at=transition_at,
                summaries=None,
                error=None,
            )

    def complete_run(
        self,
        run_id: UUID,
        summaries: Sequence[ConfigRunSummary],
        *,
        at: datetime | None = None,
    ) -> EvalRun:
        """Atomically validate final summaries and make a fully durable run immutable."""
        return self.transition_run(
            run_id,
            EvalRunStatus.COMPLETED,
            at=at,
            summaries=summaries,
        )

    def record_outcome(self, outcome: QueryOutcome) -> EvalRun:
        payload = canonical_json(outcome)
        created_at = canonical_utc(outcome.created_at, field_name="query_outcome.created_at")
        with self._session_factory.begin() as session:
            run_row = self._require_run(session, outcome.run_id)
            run = self._decode_run_row(run_row)
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
            return self._decode_outcome_row(row)

    def list_outcomes(
        self,
        run_id: UUID,
        *,
        query_id: UUID | None = None,
        limit: int = _MAX_OUTCOME_ROWS,
    ) -> list[QueryOutcome]:
        """Return a bounded deterministic durable outcome selection for one run."""
        self._validate_limit(limit, maximum=_MAX_OUTCOME_ROWS)
        with self._session_factory() as session:
            self._decode_run_row(self._require_run(session, run_id))
            statement = select(QueryOutcomeRow).where(QueryOutcomeRow.run_id == str(run_id))
            if query_id is not None:
                statement = statement.where(QueryOutcomeRow.query_id == str(query_id))
            rows = session.scalars(
                statement.order_by(QueryOutcomeRow.query_id, QueryOutcomeRow.config_id).limit(
                    limit + 1
                )
            ).all()
            if len(rows) > limit:
                raise PersistenceValidationError("durable outcome selection exceeds its bound")
            return [self._decode_outcome_row(row) for row in rows]

    def interrupt_stale_runs(self, *, at: datetime | None = None) -> list[UUID]:
        """Fail closed on work that was running when the owning process stopped."""
        transition_at = at or datetime.now(UTC)
        canonical_utc(transition_at, field_name="startup_recovery.at")
        interrupted: list[UUID] = []
        with self._session_factory.begin() as session:
            rows = session.scalars(
                select(EvalRunRow)
                .where(EvalRunRow.status == EvalRunStatus.RUNNING.value)
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
        return PufferLabRepository._decode_dataset_row(row)

    @staticmethod
    def _require_config(session: Session, config_id: UUID) -> RetrievalConfig:
        row = session.get(RetrievalConfigRow, str(config_id))
        if row is None:
            raise PersistenceValidationError(
                f"retrieval config {config_id} must be persisted first"
            )
        return PufferLabRepository._decode_config_row(row)

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
        value = PufferLabRepository._decode_query_set_row(row)
        query_rows = session.scalars(
            select(JudgedQueryRow)
            .where(JudgedQueryRow.query_set_id == row.id)
            .order_by(JudgedQueryRow.ordinal)
        ).all()
        queries = [PufferLabRepository._load_judged_query(session, row) for row in query_rows]
        if len(queries) != value.query_count:
            raise PersistenceValidationError(
                "stored judged-query count does not match the query-set payload"
            )
        return value, queries

    @staticmethod
    def _validate_document_title(value: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            raise PersistenceValidationError("judged-document title is invalid")
        return value

    @staticmethod
    def _load_judged_query(session: Session, row: JudgedQueryRow) -> JudgedQuery:
        try:
            qrels = session.scalars(
                select(QrelRow)
                .where(
                    QrelRow.query_set_id == row.query_set_id,
                    QrelRow.query_id == row.query_id,
                )
                .order_by(QrelRow.ordinal)
            ).all()
            query_data = json.loads(row.payload_json)
            query_data["qrels"] = [
                Qrel(document_id=UUID(qrel.document_id), relevance_grade=qrel.relevance_grade)
                for qrel in qrels
            ]
            query = JudgedQuery.model_validate(query_data)
        except (TypeError, ValueError):
            raise PersistenceValidationError("stored judged-query payload is invalid") from None
        if str(query.id) != row.query_id:
            raise PersistenceValidationError(
                "stored judged-query payload does not match its indexed identity"
            )
        return query

    @staticmethod
    def _decode_dataset_row(row: DatasetVersionRow) -> DatasetVersion:
        try:
            value = DatasetVersion.model_validate_json(row.payload_json)
        except (TypeError, ValueError):
            raise PersistenceValidationError("stored dataset payload is invalid") from None
        if (
            str(value.id) != row.id
            or value.slug != row.slug
            or value.version != row.version
            or value.corpus_hash != row.corpus_hash
            or canonical_utc(value.created_at, field_name="dataset_version.created_at")
            != row.created_at
        ):
            raise PersistenceValidationError(
                "stored dataset payload does not match its indexed identity"
            )
        return value

    @staticmethod
    def _decode_config_row(row: RetrievalConfigRow) -> RetrievalConfig:
        try:
            value = RetrievalConfig.model_validate_json(row.payload_json)
        except (TypeError, ValueError):
            raise PersistenceValidationError("stored retrieval-config payload is invalid") from None
        if (
            str(value.id) != row.id
            or value.revision != row.revision
            or str(value.dataset_version_id) != row.dataset_version_id
            or value.name != row.name
            or value.config_hash != row.config_hash
            or canonical_utc(value.created_at, field_name="retrieval_config.created_at")
            != row.created_at
        ):
            raise PersistenceValidationError(
                "stored retrieval-config payload does not match its indexed identity"
            )
        return value

    @staticmethod
    def _decode_query_set_row(row: QuerySetRow) -> QuerySet:
        try:
            value = QuerySet.model_validate_json(row.payload_json)
        except (TypeError, ValueError):
            raise PersistenceValidationError("stored query-set payload is invalid") from None
        if (
            str(value.id) != row.id
            or str(value.dataset_version_id) != row.dataset_version_id
            or value.name != row.name
            or value.version != row.version
            or value.content_hash != row.content_hash
            or canonical_utc(value.created_at, field_name="query_set.created_at") != row.created_at
        ):
            raise PersistenceValidationError(
                "stored query-set payload does not match its indexed identity"
            )
        return value

    @staticmethod
    def _decode_run_row(row: EvalRunRow) -> EvalRun:
        try:
            value = EvalRun.model_validate_json(row.payload_json)
        except (TypeError, ValueError):
            raise PersistenceValidationError("stored eval-run payload is invalid") from None
        started_at = (
            canonical_utc(value.started_at, field_name="eval_run.started_at")
            if value.started_at is not None
            else None
        )
        completed_at = (
            canonical_utc(value.completed_at, field_name="eval_run.completed_at")
            if value.completed_at is not None
            else None
        )
        if (
            str(value.id) != row.id
            or str(value.query_set.id) != row.query_set_id
            or value.status.value != row.status
            or value.completed_queries != row.completed_queries
            or value.total_queries != row.total_queries
            or canonical_utc(value.created_at, field_name="eval_run.created_at") != row.created_at
            or started_at != row.started_at
            or completed_at != row.completed_at
        ):
            raise PersistenceValidationError(
                "stored eval-run payload does not match its indexed lifecycle state"
            )
        return value

    @staticmethod
    def _decode_outcome_row(row: QueryOutcomeRow) -> QueryOutcome:
        try:
            value = QueryOutcome.model_validate_json(row.payload_json)
        except (TypeError, ValueError):
            raise PersistenceValidationError("stored query-outcome payload is invalid") from None
        if (
            str(value.run_id) != row.run_id
            or str(value.config_id) != row.config_id
            or str(value.query_id) != row.query_id
            or value.status.value != row.status
            or canonical_utc(value.created_at, field_name="query_outcome.created_at")
            != row.created_at
        ):
            raise PersistenceValidationError(
                "stored query-outcome payload does not match its indexed identity"
            )
        return value

    @staticmethod
    def _validate_limit(limit: int, *, maximum: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum:
            raise PersistenceValidationError(f"selection limit must be between 1 and {maximum}")

    def _transition_row(
        self,
        row: EvalRunRow,
        target: EvalRunStatus,
        *,
        transition_at: datetime,
        summaries: Sequence[ConfigRunSummary] | None,
        error: ApiErrorDetail | None,
    ) -> EvalRun:
        current = self._decode_run_row(row)
        allowed = _ALLOWED_TRANSITIONS.get(current.status, set())
        if target not in allowed:
            raise InvalidRunTransitionError(
                f"run transition {current.status.value} -> {target.value} is not allowed"
            )
        if target is EvalRunStatus.COMPLETED and current.completed_queries != current.total_queries:
            raise InvalidRunTransitionError("a run cannot complete before every query is durable")
        if target is EvalRunStatus.FAILED and error is None:
            raise PersistenceValidationError("failed runs require a safe run-level error")
        if error is not None and target is not EvalRunStatus.FAILED:
            raise PersistenceValidationError("only failed runs may store a run-level error")
        if error is not None:
            try:
                validated_error = ApiErrorDetail.model_validate(error.model_dump(mode="python"))
            except (AttributeError, TypeError, ValueError):
                raise PersistenceValidationError(
                    "failed runs require a contract-valid safe run-level error"
                ) from None
            if validated_error != error:
                raise PersistenceValidationError(
                    "failed runs require a contract-valid safe run-level error"
                )
            error = validated_error

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
    def _validate_completion_summaries(
        session: Session,
        row: EvalRunRow,
        summaries: Sequence[ConfigRunSummary] | None,
    ) -> None:
        if summaries is None:
            raise InvalidRunTransitionError("completed runs require final config summaries")

        run_config_ids = session.scalars(
            select(RunConfigRow.config_id)
            .where(RunConfigRow.run_id == row.id)
            .order_by(RunConfigRow.ordinal)
        ).all()
        summary_ids = [str(summary.config_id) for summary in summaries]
        if len(summary_ids) != len(set(summary_ids)):
            raise PersistenceValidationError("completion summaries contain duplicate config IDs")
        if summary_ids != list(run_config_ids):
            raise PersistenceValidationError(
                "completion requires exactly one ordered summary for every run config"
            )

        outcome_counts = {config_id: {"succeeded": 0, "failed": 0} for config_id in run_config_ids}
        rows = session.execute(
            select(QueryOutcomeRow.config_id, QueryOutcomeRow.status, func.count())
            .where(QueryOutcomeRow.run_id == row.id)
            .group_by(QueryOutcomeRow.config_id, QueryOutcomeRow.status)
        ).all()
        for config_id, status, count in rows:
            if config_id in outcome_counts and status in outcome_counts[config_id]:
                outcome_counts[config_id][status] = int(count)

        for summary in summaries:
            counts = outcome_counts[str(summary.config_id)]
            if (
                summary.completed_queries != counts["succeeded"]
                or summary.failed_queries != counts["failed"]
                or summary.completed_queries + summary.failed_queries != row.total_queries
            ):
                raise PersistenceValidationError(
                    "completion summary outcome counts do not match durable outcomes"
                )

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
