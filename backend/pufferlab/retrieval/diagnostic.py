"""Mode-bound retrieval composition for one expected-document diagnostic."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import NoReturn, cast
from uuid import UUID

from pufferlab.contracts.common import ScoreKind
from pufferlab.contracts.filters import FilterNode
from pufferlab.contracts.retrieval import RetrievalConfigSummary, RetrievalMode
from pufferlab.retrieval.config import SeededSearchConfig
from pufferlab.retrieval.diagnostic_types import (
    DiagnosticAttributeValue,
    DiagnosticCandidateList,
    DiagnosticCandidateRow,
    DiagnosticProvider,
    DiagnosticProviderRequest,
    DiagnosticProviderResult,
    DiagnosticTargetObservation,
    diagnostic_filter_fields,
    is_valid_diagnostic_namespace,
)
from pufferlab.retrieval.types import QueryEmbedder, QueryEmbedding

type DiagnosticProviderFactory = Callable[[], Awaitable[DiagnosticProvider]]


class DiagnosticRetrievalConfigurationError(ValueError):
    def __init__(self) -> None:
        super().__init__("expected-document diagnostic retrieval configuration is invalid")


class DiagnosticRetrievalFailure(RuntimeError):
    def __init__(self) -> None:
        super().__init__("expected-document diagnostic retrieval failed")


@dataclass(frozen=True, slots=True, repr=False)
class DiagnosticRetrievalInput:
    namespace: str
    query_text: str
    target_document_id: UUID
    config: SeededSearchConfig
    stored_filter: FilterNode | None
    include_no_filter_counterfactual: bool


@dataclass(frozen=True, slots=True, repr=False)
class DiagnosticRetrievalResult:
    provider_result: DiagnosticProviderResult
    embedding_duration_ms: float | None


@dataclass(frozen=True, slots=True)
class _EmbeddingOutcome:
    embedding: QueryEmbedding | None = None
    failed: bool = False
    cancelled: bool = False
    keyboard_interrupt: bool = False
    system_exit: bool = False


@dataclass(frozen=True, slots=True)
class _ExecutionOutcome:
    result: DiagnosticRetrievalResult | None = None
    failed: bool = False
    cancelled: bool = False
    keyboard_interrupt: bool = False
    system_exit: bool = False


async def execute_expected_document_diagnostic(
    request: DiagnosticRetrievalInput,
    *,
    provider_factory: DiagnosticProviderFactory,
    query_embedder: QueryEmbedder | None,
) -> DiagnosticRetrievalResult:
    """Validate, optionally embed, execute once, and close the request-owned provider."""

    validation_failed = False
    validation_control = (False, False, False)
    try:
        _validate_input(request, query_embedder=query_embedder)
    except BaseException as error:
        validation_control = _classify_control(error)
        validation_failed = True
        _detach_exception(error)
    if validation_failed:
        cancelled, keyboard, system = validation_control
        request = cast(DiagnosticRetrievalInput, None)
        provider_factory = cast(DiagnosticProviderFactory, None)
        query_embedder = cast(QueryEmbedder, None)
        if cancelled:
            _raise_cancelled()
        if keyboard:
            _raise_keyboard_interrupt()
        if system:
            _raise_system_exit()
        _raise_configuration_error()

    outcome = await _execute_sensitive(
        request,
        provider_factory=provider_factory,
        query_embedder=query_embedder,
    )
    request = cast(DiagnosticRetrievalInput, None)
    provider_factory = cast(DiagnosticProviderFactory, None)
    query_embedder = cast(QueryEmbedder, None)
    if outcome.cancelled:
        _raise_cancelled()
    if outcome.keyboard_interrupt:
        _raise_keyboard_interrupt()
    if outcome.system_exit:
        _raise_system_exit()
    if outcome.failed or outcome.result is None:
        _raise_failure()
    return outcome.result


def _validate_input(
    request: DiagnosticRetrievalInput,
    *,
    query_embedder: QueryEmbedder | None,
) -> None:
    if not isinstance(request, DiagnosticRetrievalInput):
        raise ValueError("diagnostic retrieval input is invalid")
    config = request.config
    if not isinstance(config, SeededSearchConfig) or not isinstance(config.mode, RetrievalMode):
        raise ValueError("diagnostic retrieval config is invalid")
    _validate_config(config)
    if (
        not is_valid_diagnostic_namespace(request.namespace)
        or not isinstance(request.query_text, str)
        or not request.query_text
        or not isinstance(request.target_document_id, UUID)
        or type(request.include_no_filter_counterfactual) is not bool
    ):
        raise ValueError("diagnostic retrieval binding is invalid")
    diagnostic_filter_fields(request.stored_filter)
    if request.include_no_filter_counterfactual and request.stored_filter is None:
        raise ValueError("diagnostic no-filter input is ineligible")

    vector = config.mode in {
        RetrievalMode.VECTOR,
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RERANK,
    }
    if vector:
        if query_embedder is None:
            raise ValueError("diagnostic query embedder is required")
        if (
            query_embedder.model != config.embedding_model
            or query_embedder.revision != config.embedding_revision
            or query_embedder.dimensions != config.embedding_dimensions
        ):
            raise ValueError("diagnostic query embedder does not match the config")
    elif query_embedder is not None:
        raise ValueError("BM25 diagnostics cannot construct a query embedder")


def _validate_config(config: SeededSearchConfig) -> None:
    summary = config.summary
    if not isinstance(summary, RetrievalConfigSummary):
        raise ValueError("diagnostic config summary is invalid")
    checked_summary = RetrievalConfigSummary.model_validate(summary.model_dump(mode="python"))
    if checked_summary != summary or checked_summary.mode is not config.mode:
        raise ValueError("diagnostic config summary is invalid")
    if (
        not isinstance(config.result_k, int)
        or isinstance(config.result_k, bool)
        or not isinstance(config.candidate_k, int)
        or isinstance(config.candidate_k, bool)
        or config.result_k != 50
        or config.candidate_k != 100
        or config.consistency != "strong"
    ):
        raise ValueError("diagnostic config bounds are invalid")
    lexical = config.mode in {
        RetrievalMode.BM25,
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RERANK,
    }
    vector = config.mode in {
        RetrievalMode.VECTOR,
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RERANK,
    }
    hybrid = config.mode in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}
    rerank = config.mode is RetrievalMode.HYBRID_RERANK
    if (config.lexical_fields is not None) is not lexical:
        raise ValueError("diagnostic lexical config is invalid")
    if lexical:
        DiagnosticProviderRequest(
            namespace="validation",
            query_text="validation",
            target_document_id=UUID(int=1),
            mode=RetrievalMode.BM25,
            lexical_fields=config.lexical_fields,
            vector_attribute=None,
            query_vector=None,
            distance_metric=None,
            stored_filter=None,
            include_no_filter_counterfactual=False,
        )
    vector_parts = (
        config.vector_attribute,
        config.embedding_model,
        config.embedding_revision,
        config.embedding_dimensions,
        config.distance_metric,
    )
    if (vector and not all(item is not None for item in vector_parts)) or (
        not vector and any(item is not None for item in vector_parts)
    ):
        raise ValueError("diagnostic vector config is invalid")
    if vector and (
        not isinstance(config.vector_attribute, str)
        or not config.vector_attribute
        or not isinstance(config.embedding_model, str)
        or not config.embedding_model
        or not isinstance(config.embedding_revision, str)
        or not config.embedding_revision
        or not isinstance(config.embedding_dimensions, int)
        or isinstance(config.embedding_dimensions, bool)
        or config.embedding_dimensions < 1
        or config.distance_metric not in {"cosine_distance", "euclidean_squared"}
    ):
        raise ValueError("diagnostic vector config is invalid")
    rrf_parts = (config.rrf_rank_constant, config.rrf_weights)
    if (hybrid and not all(item is not None for item in rrf_parts)) or (
        not hybrid and any(item is not None for item in rrf_parts)
    ):
        raise ValueError("diagnostic RRF config is invalid")
    if hybrid and (
        not isinstance(config.rrf_rank_constant, int)
        or isinstance(config.rrf_rank_constant, bool)
        or config.rrf_rank_constant < 1
        or config.rrf_rank_constant > 10_000
        or not isinstance(config.rrf_weights, tuple)
        or len(config.rrf_weights) != 2
        or any(
            not isinstance(weight, int | float)
            or isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight <= 0
            or weight > 100
            for weight in config.rrf_weights
        )
    ):
        raise ValueError("diagnostic RRF config is invalid")
    reranker_parts = (config.reranker_model, config.reranker_revision, config.reranker_depth)
    if (rerank and not all(item is not None for item in reranker_parts)) or (
        not rerank and any(item is not None for item in reranker_parts)
    ):
        raise ValueError("diagnostic reranker config is invalid")
    if rerank and (
        not isinstance(config.reranker_model, str)
        or not config.reranker_model
        or not isinstance(config.reranker_revision, str)
        or not config.reranker_revision
        or not isinstance(config.reranker_depth, int)
        or isinstance(config.reranker_depth, bool)
        or config.reranker_depth != 50
    ):
        raise ValueError("diagnostic reranker config is invalid")


async def _execute_sensitive(
    request: DiagnosticRetrievalInput,
    *,
    provider_factory: DiagnosticProviderFactory,
    query_embedder: QueryEmbedder | None,
) -> _ExecutionOutcome:
    embedding_outcome = await _embed_if_required(request, query_embedder=query_embedder)
    if (
        embedding_outcome.failed
        or embedding_outcome.cancelled
        or embedding_outcome.keyboard_interrupt
        or embedding_outcome.system_exit
    ):
        return _ExecutionOutcome(
            failed=embedding_outcome.failed,
            cancelled=embedding_outcome.cancelled,
            keyboard_interrupt=embedding_outcome.keyboard_interrupt,
            system_exit=embedding_outcome.system_exit,
        )

    embedding = embedding_outcome.embedding
    query_vector = None if embedding is None else embedding.vector
    embedding_duration_ms = None if embedding is None else embedding.client_duration_ms
    embedding_outcome = cast(_EmbeddingOutcome, None)
    provider: DiagnosticProvider | None = None
    result: DiagnosticProviderResult | None = None
    operation_error: BaseException | None = None
    try:
        provider_request = DiagnosticProviderRequest(
            namespace=request.namespace,
            query_text=request.query_text,
            target_document_id=request.target_document_id,
            mode=request.config.mode,
            lexical_fields=request.config.lexical_fields,
            vector_attribute=request.config.vector_attribute,
            query_vector=query_vector,
            distance_metric=request.config.distance_metric,
            stored_filter=request.stored_filter,
            include_no_filter_counterfactual=request.include_no_filter_counterfactual,
            result_k=request.config.result_k,
            candidate_k=request.config.candidate_k,
        )
        provider = await provider_factory()
        result = _validated_provider_result(
            await provider.query(provider_request),
            request=provider_request,
        )
    except BaseException as error:
        operation_error = error
        _detach_exception(error)

    request = cast(DiagnosticRetrievalInput, None)
    provider_factory = cast(DiagnosticProviderFactory, None)
    query_embedder = cast(QueryEmbedder, None)
    provider_request = cast(DiagnosticProviderRequest, None)
    embedding = None
    query_vector = None

    close_error: BaseException | None = None
    close_cancelled = False
    if provider is not None:
        close_error, close_cancelled = await _drain_close(provider)
    provider = None

    cancelled, keyboard, system = _classify_control_optional(operation_error)
    operation_failed = operation_error is not None and not (cancelled or keyboard or system)
    operation_error = None
    close_control = _classify_control_optional(close_error)
    close_failed = close_error is not None and not any(close_control)
    close_error = None
    if not (cancelled or keyboard or system):
        if close_cancelled:
            cancelled = True
        else:
            cancelled, keyboard, system = close_control
    if operation_failed or close_failed:
        result = None
    if cancelled or keyboard or system:
        result = None
    if result is None:
        return _ExecutionOutcome(
            failed=operation_failed or close_failed or not (cancelled or keyboard or system),
            cancelled=cancelled,
            keyboard_interrupt=keyboard,
            system_exit=system,
        )
    return _ExecutionOutcome(
        result=DiagnosticRetrievalResult(
            provider_result=result,
            embedding_duration_ms=embedding_duration_ms,
        )
    )


async def _embed_if_required(
    request: DiagnosticRetrievalInput,
    *,
    query_embedder: QueryEmbedder | None,
) -> _EmbeddingOutcome:
    if query_embedder is None:
        return _EmbeddingOutcome()
    expected_dimensions = request.config.embedding_dimensions
    embedding_operation: object | None = None
    captured_operation: object | None = None
    try:
        embedding_operation = query_embedder.embed_query(request.query_text)
        captured_operation = _capture_embedding(
            cast(Awaitable[QueryEmbedding], embedding_operation)
        )
        task = asyncio.create_task(captured_operation)
    except BaseException as caught_start:
        if captured_operation is not None:
            _dispose_unstarted_awaitable(captured_operation)
        if embedding_operation is not None:
            _dispose_unstarted_awaitable(embedding_operation)
        control = _classify_control(caught_start)
        _detach_exception(caught_start)
        request = cast(DiagnosticRetrievalInput, None)
        query_embedder = cast(QueryEmbedder, None)
        return _EmbeddingOutcome(
            failed=not any(control),
            cancelled=control[0],
            keyboard_interrupt=control[1],
            system_exit=control[2],
        )
    cancelled = False
    captured: tuple[QueryEmbedding | None, BaseException | None] = (None, None)
    captured_ready = False
    while True:
        try:
            captured = await asyncio.shield(task)
            captured_ready = True
            break
        except asyncio.CancelledError:
            cancelled = True
            if task.done():
                break
    if task.done() and not captured_ready:
        captured = task.result()
    task = cast(asyncio.Task[tuple[QueryEmbedding | None, BaseException | None]], None)
    request = cast(DiagnosticRetrievalInput, None)
    query_embedder = cast(QueryEmbedder, None)
    embedding, error = captured
    if cancelled:
        embedding = None
        error = None
        return _EmbeddingOutcome(cancelled=True)
    if error is not None:
        control = _classify_control(error)
        error = None
        embedding = None
        return _EmbeddingOutcome(
            failed=not any(control),
            cancelled=control[0],
            keyboard_interrupt=control[1],
            system_exit=control[2],
        )
    if (
        not isinstance(expected_dimensions, int)
        or isinstance(expected_dimensions, bool)
        or embedding is None
        or not _valid_embedding(embedding, expected_dimensions=expected_dimensions)
    ):
        embedding = None
        return _EmbeddingOutcome(failed=True)
    return _EmbeddingOutcome(embedding=embedding)


async def _capture_embedding(
    operation: Awaitable[QueryEmbedding],
) -> tuple[QueryEmbedding | None, BaseException | None]:
    try:
        return await operation, None
    except BaseException as error:
        _detach_exception(error)
        return None, error


def _valid_embedding(embedding: QueryEmbedding, *, expected_dimensions: int | None) -> bool:
    try:
        return (
            isinstance(embedding, QueryEmbedding)
            and isinstance(embedding.vector, tuple)
            and bool(embedding.vector)
            and (expected_dimensions is None or len(embedding.vector) == expected_dimensions)
            and all(
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in embedding.vector
            )
            and isinstance(embedding.client_duration_ms, int | float)
            and not isinstance(embedding.client_duration_ms, bool)
            and math.isfinite(embedding.client_duration_ms)
            and embedding.client_duration_ms >= 0
            and embedding.client_duration_ms <= 600_000.0
        )
    except (OverflowError, TypeError, ValueError):
        return False


async def _drain_close(provider: DiagnosticProvider) -> tuple[BaseException | None, bool]:
    close_operation: object | None = None
    captured_operation: object | None = None
    start_error: BaseException | None = None
    try:
        close_operation = provider.close()
        captured_operation = _capture_close(cast(Awaitable[None], close_operation))
        task = asyncio.create_task(captured_operation)
    except BaseException as caught_start:
        _detach_exception(caught_start)
        start_error = caught_start
        if captured_operation is None:
            if close_operation is not None:
                _dispose_unstarted_awaitable(close_operation)
            return start_error, False
        try:
            task = asyncio.ensure_future(cast(Awaitable[BaseException | None], captured_operation))
        except BaseException as caught_fallback:
            _detach_exception(caught_fallback)
            _dispose_unstarted_awaitable(captured_operation)
            fallback_error = await _capture_close(cast(Awaitable[None], close_operation))
            control = _classify_control_optional(fallback_error)
            if any(control):
                return fallback_error, False
            return start_error, False
    cancelled = False
    error: BaseException | None = None
    captured_ready = False
    while True:
        try:
            error = await asyncio.shield(task)
            captured_ready = True
            break
        except asyncio.CancelledError:
            cancelled = True
            if task.done():
                break
    if task.done() and not captured_ready:
        error = task.result()
    task = cast(asyncio.Task[BaseException | None], None)
    provider = cast(DiagnosticProvider, None)
    if start_error is not None and error is None and not cancelled:
        error = start_error
    return error, cancelled


async def _capture_close(operation: Awaitable[None]) -> BaseException | None:
    try:
        await operation
    except BaseException as error:
        _detach_exception(error)
        return error
    return None


def _validated_provider_result(
    value: object,
    *,
    request: DiagnosticProviderRequest,
) -> DiagnosticProviderResult:
    if type(value) is not DiagnosticProviderResult:
        raise ValueError("diagnostic provider returned an invalid result")
    target = value.target
    if type(target) is not DiagnosticTargetObservation:
        raise ValueError("diagnostic provider returned an invalid target")
    if type(target.attributes) is not tuple or len(target.attributes) > 16:
        raise ValueError("diagnostic provider returned invalid target attributes")
    if any(type(item) is not DiagnosticAttributeValue for item in target.attributes):
        raise ValueError("diagnostic provider returned invalid target attributes")
    candidates = value.candidate_lists
    if type(candidates) is not tuple or len(candidates) != len(request.roles) - 1:
        raise ValueError("diagnostic provider returned invalid candidate containers")
    for candidate in candidates:
        if type(candidate) is not DiagnosticCandidateList:
            raise ValueError("diagnostic provider returned invalid candidate containers")
        if type(candidate.rows) is not tuple or len(candidate.rows) > request.candidate_limit:
            raise ValueError("diagnostic provider returned invalid candidate rows")
        if any(type(row) is not DiagnosticCandidateRow for row in candidate.rows):
            raise ValueError("diagnostic provider returned invalid candidate rows")
    validated_target = DiagnosticTargetObservation(
        target_document_id=target.target_document_id,
        available=target.available,
        bm25_score=target.bm25_score,
        vector_distance=target.vector_distance,
        attributes=tuple(
            DiagnosticAttributeValue(field=item.field, state=item.state, value=item.value)
            for item in target.attributes
        ),
    )
    validated_lists = tuple(
        DiagnosticCandidateList(
            role=candidate.role,
            requested_limit=candidate.requested_limit,
            rows=tuple(
                DiagnosticCandidateRow(
                    document_id=row.document_id,
                    rank=row.rank,
                    score=row.score,
                )
                for row in candidate.rows
            ),
        )
        for candidate in candidates
    )
    validated = DiagnosticProviderResult(
        namespace=value.namespace,
        target=validated_target,
        candidate_lists=validated_lists,
        client_duration_ms=value.client_duration_ms,
    )
    if validated.namespace != request.namespace:
        raise ValueError("diagnostic result namespace does not match the request")
    if validated.target.available:
        if (validated.target.bm25_score is not None) is (request.lexical_fields is None):
            raise ValueError("diagnostic target lexical score does not match the request")
        if (validated.target.vector_distance is not None) is (request.query_vector is None):
            raise ValueError("diagnostic target vector score does not match the request")
        if tuple(item.field for item in validated.target.attributes) != request.filter_fields:
            raise ValueError("diagnostic target attributes do not match the request")
    if validated.target.target_document_id != request.target_document_id:
        raise ValueError("diagnostic target identity does not match the request")
    if tuple(item.role for item in validated.candidate_lists) != request.roles[1:]:
        raise ValueError("diagnostic candidate roles do not match the request")
    if any(item.requested_limit != request.candidate_limit for item in validated.candidate_lists):
        raise ValueError("diagnostic candidate limits do not match the request")
    for candidate in validated.candidate_lists:
        matching = next(
            (row for row in candidate.rows if row.document_id == request.target_document_id),
            None,
        )
        if matching is None:
            continue
        if not validated.target.available:
            raise ValueError("diagnostic candidate contradicts unavailable target")
        direct = (
            validated.target.bm25_score
            if matching.score.kind is ScoreKind.BM25
            else validated.target.vector_distance
        )
        if direct is None or not math.isclose(
            matching.score.value,
            direct.value,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("diagnostic candidate score contradicts the target lookup")
    return validated


def _detach_exception(error: BaseException) -> None:
    error.__traceback__ = None
    error.__cause__ = None
    error.__context__ = None


def _dispose_unstarted_awaitable(value: object) -> None:
    if inspect.iscoroutine(value):
        value.close()
    elif isinstance(value, asyncio.Future) and not value.done():
        value.cancel()


def _classify_control(error: BaseException) -> tuple[bool, bool, bool]:
    return (
        isinstance(error, asyncio.CancelledError),
        isinstance(error, KeyboardInterrupt),
        isinstance(error, SystemExit),
    )


def _classify_control_optional(error: BaseException | None) -> tuple[bool, bool, bool]:
    return (False, False, False) if error is None else _classify_control(error)


def _raise_configuration_error() -> NoReturn:
    raise DiagnosticRetrievalConfigurationError()


def _raise_failure() -> NoReturn:
    raise DiagnosticRetrievalFailure()


def _raise_cancelled() -> NoReturn:
    raise asyncio.CancelledError()


def _raise_keyboard_interrupt() -> NoReturn:
    raise KeyboardInterrupt()


def _raise_system_exit() -> NoReturn:
    raise SystemExit(1)
