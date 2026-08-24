from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from typing import cast
from uuid import UUID

import pufferlab.retrieval.diagnostic as diagnostic_module
import pytest
from pufferlab.contracts.common import (
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.filters import (
    FilterLogical,
    FilterPredicate,
    LogicalOp,
    PredicateOp,
)
from pufferlab.contracts.forensics import DiagnosticSubqueryRole
from pufferlab.contracts.retrieval import RetrievalConfigSummary, RetrievalMode
from pufferlab.retrieval.config import SeededSearchConfig
from pufferlab.retrieval.diagnostic import (
    DiagnosticRetrievalConfigurationError,
    DiagnosticRetrievalFailure,
    DiagnosticRetrievalInput,
    execute_expected_document_diagnostic,
)
from pufferlab.retrieval.diagnostic_types import (
    DiagnosticAttributeState,
    DiagnosticAttributeValue,
    DiagnosticCandidateList,
    DiagnosticCandidateRow,
    DiagnosticProviderRequest,
    DiagnosticProviderResult,
    DiagnosticTargetObservation,
    diagnostic_filter_fields,
)
from pufferlab.retrieval.types import QueryEmbedding

_CONFIG_ID = UUID("10000000-0000-0000-0000-000000000001")
_TARGET = UUID("20000000-0000-0000-0000-000000000001")
_OTHER = UUID("20000000-0000-0000-0000-000000000002")
_MARKER = "sensitive-query-filter-provider-marker"


def _config(mode: RetrievalMode) -> SeededSearchConfig:
    lexical = mode in {
        RetrievalMode.BM25,
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RERANK,
    }
    vector = mode in {
        RetrievalMode.VECTOR,
        RetrievalMode.HYBRID_RRF,
        RetrievalMode.HYBRID_RERANK,
    }
    return SeededSearchConfig(
        summary=RetrievalConfigSummary(
            id=_CONFIG_ID,
            revision=1,
            name=mode.value,
            mode=mode,
            config_hash="config-hash",
        ),
        mode=mode,
        result_k=50,
        candidate_k=100,
        consistency="strong",
        lexical_fields=(("title", 2.0), ("body", 1.0)) if lexical else None,
        vector_attribute="vector" if vector else None,
        embedding_model="fixture-model" if vector else None,
        embedding_revision="fixture-revision" if vector else None,
        embedding_dimensions=2 if vector else None,
        distance_metric="cosine_distance" if vector else None,
        rrf_rank_constant=60
        if mode in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}
        else None,
        rrf_weights=(1.0, 1.0)
        if mode in {RetrievalMode.HYBRID_RRF, RetrievalMode.HYBRID_RERANK}
        else None,
        reranker_model="fixture-reranker" if mode is RetrievalMode.HYBRID_RERANK else None,
        reranker_revision="fixture-reranker-revision"
        if mode is RetrievalMode.HYBRID_RERANK
        else None,
        reranker_depth=50 if mode is RetrievalMode.HYBRID_RERANK else None,
    )


def _input(
    mode: RetrievalMode, *, stored_filter: FilterPredicate | None = None
) -> DiagnosticRetrievalInput:
    return DiagnosticRetrievalInput(
        namespace="m5_fixture",
        query_text=_MARKER,
        target_document_id=_TARGET,
        config=_config(mode),
        stored_filter=stored_filter,
        include_no_filter_counterfactual=False,
    )


def _score(kind: ScoreKind, value: float, *, direct: bool) -> ObservedScore:
    return ObservedScore(
        kind=kind,
        value=value,
        direction=(
            ScoreDirection.LOWER_IS_BETTER
            if kind is ScoreKind.VECTOR_DISTANCE
            else ScoreDirection.HIGHER_IS_BETTER
        ),
        source=ScoreSource.COMPUTE_ATTRIBUTE if direct else ScoreSource.TURBOPUFFER_DIST,
    )


def _result(request: DiagnosticProviderRequest) -> DiagnosticProviderResult:
    target = DiagnosticTargetObservation(
        target_document_id=_TARGET,
        available=True,
        bm25_score=_score(ScoreKind.BM25, 4.0, direct=True)
        if request.lexical_fields is not None
        else None,
        vector_distance=_score(ScoreKind.VECTOR_DISTANCE, 0.25, direct=True)
        if request.query_vector is not None
        else None,
        attributes=(),
    )
    candidates = tuple(
        DiagnosticCandidateList(
            role=role,
            requested_limit=request.candidate_limit,
            rows=(
                DiagnosticCandidateRow(
                    document_id=_TARGET,
                    rank=1,
                    score=_score(
                        ScoreKind.BM25 if "bm25" in role.value else ScoreKind.VECTOR_DISTANCE,
                        4.0 if "bm25" in role.value else 0.25,
                        direct=False,
                    ),
                ),
            ),
        )
        for role in request.roles[1:]
    )
    return DiagnosticProviderResult(
        namespace=request.namespace,
        target=target,
        candidate_lists=candidates,
        client_duration_ms=3.0,
    )


class FakeProvider:
    def __init__(
        self,
        *,
        returned: object | None = None,
        query_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.returned = returned
        self.query_error = query_error
        self.close_error = close_error
        self.query_calls = 0
        self.close_calls = 0
        self.request: DiagnosticProviderRequest | None = None

    async def query(self, request: DiagnosticProviderRequest) -> DiagnosticProviderResult:
        self.query_calls += 1
        self.request = request
        if self.query_error is not None:
            raise self.query_error
        if self.returned is None:
            return _result(request)
        return cast(DiagnosticProviderResult, self.returned)

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class FakeEmbedder:
    model = "fixture-model"
    revision = "fixture-revision"
    dimensions = 2

    def __init__(self, embedding: QueryEmbedding | None = None) -> None:
        self.embedding = embedding or QueryEmbedding(vector=(0.25, -0.5), client_duration_ms=2.0)
        self.calls = 0

    async def embed_query(self, query_text: str) -> QueryEmbedding:
        assert query_text == _MARKER
        self.calls += 1
        return self.embedding


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(RetrievalMode))
async def test_mode_composition_embeds_at_most_once_constructs_provider_afterward_and_closes_once(
    mode: RetrievalMode,
) -> None:
    events: list[str] = []
    provider = FakeProvider()
    embedder = FakeEmbedder() if mode is not RetrievalMode.BM25 else None

    async def factory() -> FakeProvider:
        events.append("factory")
        return provider

    if embedder is not None:
        original = embedder.embed_query

        async def recording_embed(query_text: str) -> QueryEmbedding:
            events.append("embed")
            return await original(query_text)

        embedder.embed_query = recording_embed  # type: ignore[method-assign]

    result = await execute_expected_document_diagnostic(
        _input(mode),
        provider_factory=factory,
        query_embedder=embedder,
    )

    assert events == (["factory"] if embedder is None else ["embed", "factory"])
    assert provider.query_calls == 1
    assert provider.close_calls == 1
    assert (result.embedding_duration_ms is None) is (embedder is None)
    assert provider.request is not None
    assert provider.request.roles[0] is DiagnosticSubqueryRole.TARGET_LOOKUP
    assert len(provider.request.roles) == (
        2 if mode in {RetrievalMode.BM25, RetrievalMode.VECTOR} else 3
    )


@pytest.mark.asyncio
async def test_invalid_constructed_filter_rejects_before_embedder_or_provider() -> None:
    invalid = FilterPredicate.model_construct(
        kind="predicate",
        field="unsafe field",
        op=PredicateOp.EQ,
        value=_MARKER,
    )
    embedder = FakeEmbedder()
    factory_calls = 0

    async def factory() -> FakeProvider:
        nonlocal factory_calls
        factory_calls += 1
        return FakeProvider()

    with pytest.raises(DiagnosticRetrievalConfigurationError) as raised:
        await execute_expected_document_diagnostic(
            _input(RetrievalMode.VECTOR, stored_filter=invalid),
            provider_factory=factory,
            query_embedder=embedder,
        )

    assert embedder.calls == 0
    assert factory_calls == 0
    assert _MARKER not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "poison"),
    [
        (RetrievalMode.BM25, "summary"),
        (RetrievalMode.BM25, "irrelevant_rrf"),
        (RetrievalMode.VECTOR, "irrelevant_lexical"),
        (RetrievalMode.HYBRID_RRF, "missing_rrf"),
        (RetrievalMode.HYBRID_RRF, "rank_constant_overflow"),
        (RetrievalMode.HYBRID_RRF, "weight_overflow"),
        (RetrievalMode.HYBRID_RERANK, "missing_reranker"),
    ],
)
async def test_complete_mode_config_is_validated_before_embedding_or_provider(
    mode: RetrievalMode,
    poison: str,
) -> None:
    config = _config(mode)
    if poison == "summary":
        config = replace(config, summary=cast(RetrievalConfigSummary, object()))
    elif poison == "irrelevant_rrf":
        config = replace(config, rrf_rank_constant=60, rrf_weights=(1.0, 1.0))
    elif poison == "irrelevant_lexical":
        config = replace(config, lexical_fields=(("title", 1.0),))
    elif poison == "missing_rrf":
        config = replace(config, rrf_weights=None)
    elif poison == "rank_constant_overflow":
        config = replace(config, rrf_rank_constant=10_001)
    elif poison == "weight_overflow":
        config = replace(config, rrf_weights=(100.001, 1.0))
    elif poison == "missing_reranker":
        config = replace(config, reranker_depth=None)
    request = replace(_input(mode), config=config)
    embedder = FakeEmbedder() if mode is not RetrievalMode.BM25 else None
    factory_calls = 0

    async def factory() -> FakeProvider:
        nonlocal factory_calls
        factory_calls += 1
        return FakeProvider()

    with pytest.raises(DiagnosticRetrievalConfigurationError):
        await execute_expected_document_diagnostic(
            request,
            provider_factory=factory,
            query_embedder=embedder,
        )
    assert factory_calls == 0
    if embedder is not None:
        assert embedder.calls == 0


@pytest.mark.parametrize(
    "node",
    [
        FilterPredicate.model_construct(
            kind="predicate", field="bad field", op=PredicateOp.EQ, value=1
        ),
        FilterPredicate.model_construct(kind="predicate", field="safe", op="eq", value=1),
        FilterPredicate.model_construct(
            kind="predicate", field="safe", op=PredicateOp.IN, value={}
        ),
        FilterPredicate.model_construct(
            kind="predicate", field="safe", op=PredicateOp.IN, value=[1, "1"]
        ),
        FilterLogical.model_construct(kind="logical", op=LogicalOp.NOT, children=[]),
        FilterLogical.model_construct(
            kind="logical",
            op=LogicalOp.NOT,
            children=[
                FilterPredicate(field="safe", op=PredicateOp.EQ, value=1),
                FilterPredicate(field="safe", op=PredicateOp.EQ, value=2),
            ],
        ),
        FilterLogical.model_construct(kind="logical", op="and", children=[]),
    ],
)
def test_constructed_filter_shape_attacks_are_bounded_value_errors(node: object) -> None:
    with pytest.raises(ValueError):
        diagnostic_filter_fields(cast(FilterPredicate, node))


@pytest.mark.parametrize(
    "field",
    ["id", "__pufferlab_diagnostic_bm25", "__pufferlab_diagnostic_vector_distance"],
)
def test_provider_and_compute_alias_filter_fields_are_reserved(field: str) -> None:
    with pytest.raises(ValueError):
        diagnostic_filter_fields(FilterPredicate(field=field, op=PredicateOp.EQ, value=1))


def test_empty_and_ten_thousand_value_filter_arrays_are_valid_but_larger_is_bounded() -> None:
    assert diagnostic_filter_fields(FilterPredicate(field="tag", op=PredicateOp.IN, value=[])) == (
        "tag",
    )
    assert diagnostic_filter_fields(
        FilterPredicate(field="tag", op=PredicateOp.IN, value=list(range(10_000)))
    ) == ("tag",)
    assert diagnostic_filter_fields(
        FilterPredicate(field="numeric", op=PredicateOp.IN, value=[1, 1.5])
    ) == ("numeric",)
    with pytest.raises(ValueError):
        diagnostic_filter_fields(FilterPredicate(field="tag", op=PredicateOp.IN, value=[None]))
    with pytest.raises(ValueError):
        diagnostic_filter_fields(
            FilterPredicate(field="tag", op=PredicateOp.IN, value=list(range(10_001)))
        )


@pytest.mark.parametrize("missing", ["kind", "value", "direction", "source"])
def test_constructed_scores_cannot_enter_target_or_candidate_rows(missing: str) -> None:
    values: dict[str, object] = {
        "kind": ScoreKind.BM25,
        "value": 1.0,
        "direction": ScoreDirection.HIGHER_IS_BETTER,
        "source": ScoreSource.TURBOPUFFER_DIST,
    }
    values.pop(missing)
    score = ObservedScore.model_construct(**values)
    with pytest.raises(ValueError):
        DiagnosticCandidateRow(document_id=_OTHER, rank=1, score=score)
    direct_values = {
        "kind": ScoreKind.BM25,
        "value": 1.0,
        "direction": ScoreDirection.HIGHER_IS_BETTER,
        "source": ScoreSource.COMPUTE_ATTRIBUTE,
    }
    direct_values.pop(missing)
    direct = ObservedScore.model_construct(**direct_values)
    with pytest.raises(ValueError):
        DiagnosticTargetObservation(
            target_document_id=_TARGET,
            available=True,
            bm25_score=direct,
            vector_distance=None,
            attributes=(),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"nested": float("nan")}])
def test_attribute_values_reject_nonfinite_json(value: object) -> None:
    with pytest.raises(ValueError):
        DiagnosticAttributeValue(
            field="attribute",
            state=DiagnosticAttributeState.PRESENT_VALUE,
            value=cast(object, value),
        )


@pytest.mark.asyncio
async def test_malformed_fake_provider_result_is_discarded_and_provider_closes_once() -> None:
    provider = FakeProvider(returned=object())

    async def factory() -> FakeProvider:
        return provider

    with pytest.raises(DiagnosticRetrievalFailure) as raised:
        await execute_expected_document_diagnostic(
            _input(RetrievalMode.BM25),
            provider_factory=factory,
            query_embedder=None,
        )

    assert provider.query_calls == 1
    assert provider.close_calls == 1
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_candidates",
        "wrong_role",
        "wrong_limit",
        "missing_direct_score",
        "attribute_mismatch",
        "unavailable_candidate",
        "target_score_mismatch",
        "swapped_target",
        "swapped_namespace",
    ],
)
async def test_fake_result_must_exactly_match_request_mode_roles_and_target(
    mutation: str,
) -> None:
    provider = FakeProvider()

    async def factory() -> FakeProvider:
        return provider

    original_query = provider.query

    async def malicious_query(request: DiagnosticProviderRequest) -> DiagnosticProviderResult:
        valid = await original_query(request)
        if mutation == "missing_candidates":
            return replace(valid, candidate_lists=())
        candidate = valid.candidate_lists[0]
        if mutation == "wrong_role":
            return replace(
                valid,
                candidate_lists=(
                    replace(
                        candidate,
                        role=DiagnosticSubqueryRole.NO_FILTER_COUNTERFACTUAL_BM25_CANDIDATES,
                    ),
                ),
            )
        if mutation == "wrong_limit":
            return replace(valid, candidate_lists=(replace(candidate, requested_limit=100),))
        if mutation == "missing_direct_score":
            return replace(valid, target=replace(valid.target, bm25_score=None))
        if mutation == "attribute_mismatch":
            return replace(
                valid,
                target=replace(
                    valid.target,
                    attributes=(
                        DiagnosticAttributeValue(
                            field="foreign",
                            state=DiagnosticAttributeState.PRESENT_VALUE,
                            value="value",
                        ),
                    ),
                ),
            )
        if mutation == "unavailable_candidate":
            return replace(
                valid,
                target=DiagnosticTargetObservation(
                    target_document_id=_TARGET,
                    available=False,
                    bm25_score=None,
                    vector_distance=None,
                    attributes=(),
                ),
            )
        if mutation == "swapped_target":
            return replace(valid, target=replace(valid.target, target_document_id=_OTHER))
        if mutation == "swapped_namespace":
            return replace(valid, namespace="foreign_namespace")
        return replace(
            valid,
            candidate_lists=(
                replace(
                    candidate,
                    rows=(
                        replace(candidate.rows[0], score=_score(ScoreKind.BM25, 3.0, direct=False)),
                    ),
                ),
            ),
        )

    provider.query = malicious_query  # type: ignore[method-assign]
    with pytest.raises(DiagnosticRetrievalFailure):
        await execute_expected_document_diagnostic(
            _input(RetrievalMode.BM25),
            provider_factory=factory,
            query_embedder=None,
        )
    assert provider.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attack",
    [
        "result_subclass",
        "missing_result_fields",
        "candidate_list_container",
        "candidate_hostile_iterable",
        "target_attribute_container",
        "candidate_row_container",
        "candidate_subclass",
        "attribute_subclass",
        "candidate_list_overbound",
        "target_attribute_overbound",
        "candidate_rows_overbound",
    ],
)
async def test_forged_fake_result_containers_reject_without_consuming_hostile_iterables(
    attack: str,
) -> None:
    provider = FakeProvider()
    hostile_consumed = False

    class HostileIterable:
        def __iter__(self) -> object:
            nonlocal hostile_consumed
            hostile_consumed = True
            raise AssertionError(_MARKER)

        def __len__(self) -> int:
            nonlocal hostile_consumed
            hostile_consumed = True
            raise AssertionError(_MARKER)

    class ResultSubclass(DiagnosticProviderResult):
        pass

    class CandidateSubclass(DiagnosticCandidateList):
        pass

    class AttributeSubclass(DiagnosticAttributeValue):
        pass

    original_query = provider.query

    async def malicious_query(request: DiagnosticProviderRequest) -> DiagnosticProviderResult:
        valid = await original_query(request)
        candidate = valid.candidate_lists[0]
        if attack == "result_subclass":
            return ResultSubclass(
                namespace=valid.namespace,
                target=valid.target,
                candidate_lists=valid.candidate_lists,
                client_duration_ms=valid.client_duration_ms,
            )
        if attack == "missing_result_fields":
            return cast(DiagnosticProviderResult, object.__new__(DiagnosticProviderResult))
        if attack == "candidate_list_container":
            object.__setattr__(valid, "candidate_lists", list(valid.candidate_lists))
        elif attack == "candidate_hostile_iterable":
            object.__setattr__(valid, "candidate_lists", HostileIterable())
        elif attack == "target_attribute_container":
            object.__setattr__(valid.target, "attributes", [])
        elif attack == "candidate_row_container":
            object.__setattr__(candidate, "rows", list(candidate.rows))
        elif attack == "candidate_subclass":
            object.__setattr__(
                valid,
                "candidate_lists",
                (
                    CandidateSubclass(
                        role=candidate.role,
                        requested_limit=candidate.requested_limit,
                        rows=candidate.rows,
                    ),
                ),
            )
        elif attack == "attribute_subclass":
            object.__setattr__(
                valid.target,
                "attributes",
                (
                    AttributeSubclass(
                        field="category",
                        state=DiagnosticAttributeState.PRESENT_VALUE,
                        value="value",
                    ),
                ),
            )
        elif attack == "candidate_list_overbound":
            object.__setattr__(valid, "candidate_lists", (candidate, candidate))
        elif attack == "target_attribute_overbound":
            attribute = DiagnosticAttributeValue(
                field="category",
                state=DiagnosticAttributeState.PRESENT_VALUE,
                value="value",
            )
            object.__setattr__(valid.target, "attributes", (attribute,) * 17)
        elif attack == "candidate_rows_overbound":
            object.__setattr__(candidate, "rows", candidate.rows * 51)
        return valid

    provider.query = malicious_query  # type: ignore[method-assign]

    async def factory() -> FakeProvider:
        return provider

    with pytest.raises(DiagnosticRetrievalFailure) as raised:
        await execute_expected_document_diagnostic(
            _input(RetrievalMode.BM25),
            provider_factory=factory,
            query_embedder=None,
        )

    assert provider.close_calls == 1
    assert hostile_consumed is False
    assert _MARKER not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_close_failure_discards_valid_result_and_is_redacted() -> None:
    provider = FakeProvider(close_error=RuntimeError(_MARKER))

    async def factory() -> FakeProvider:
        return provider

    with pytest.raises(DiagnosticRetrievalFailure) as raised:
        await execute_expected_document_diagnostic(
            _input(RetrievalMode.BM25),
            provider_factory=factory,
            query_embedder=None,
        )

    assert provider.close_calls == 1
    assert _MARKER not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation_error", "expected"),
    [
        (RuntimeError(_MARKER), DiagnosticRetrievalFailure),
        (asyncio.CancelledError(_MARKER), asyncio.CancelledError),
        (KeyboardInterrupt(_MARKER), KeyboardInterrupt),
        (SystemExit(_MARKER), SystemExit),
    ],
)
async def test_provider_operation_controls_close_once_and_reraise_fresh(
    operation_error: BaseException,
    expected: type[BaseException],
) -> None:
    provider = FakeProvider(query_error=operation_error)

    async def factory() -> FakeProvider:
        return provider

    with pytest.raises(expected) as raised:
        await execute_expected_document_diagnostic(
            _input(RetrievalMode.BM25),
            provider_factory=factory,
            query_embedder=None,
        )

    assert provider.query_calls == 1
    assert provider.close_calls == 1
    assert _MARKER not in str(raised.value)
    assert raised.value is not operation_error
    assert operation_error.__traceback__ is None
    assert operation_error.__cause__ is None
    assert operation_error.__context__ is None
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("close_error", [KeyboardInterrupt(_MARKER), SystemExit(_MARKER)])
async def test_close_controls_discard_result_and_reraise_fresh(
    close_error: BaseException,
) -> None:
    provider = FakeProvider(close_error=close_error)

    async def factory() -> FakeProvider:
        return provider

    with pytest.raises(type(close_error)) as raised:
        await execute_expected_document_diagnostic(
            _input(RetrievalMode.BM25),
            provider_factory=factory,
            query_embedder=None,
        )

    assert provider.close_calls == 1
    assert raised.value is not close_error
    assert _MARKER not in str(raised.value)
    assert close_error.__traceback__ is None
    assert close_error.__cause__ is None
    assert close_error.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_control", [KeyboardInterrupt(_MARKER), SystemExit(_MARKER)])
@pytest.mark.parametrize("cancel_count", [1, 2])
async def test_original_operation_control_wins_over_close_cancellation_after_drain(
    operation_control: BaseException,
    cancel_count: int,
) -> None:
    owner = asyncio.current_task()
    assert owner is not None

    class CancellingCloseProvider(FakeProvider):
        async def close(self) -> None:
            self.close_calls += 1
            for _ in range(cancel_count):
                owner.cancel()
            await asyncio.sleep(0)

    provider = CancellingCloseProvider(query_error=operation_control)

    async def factory() -> FakeProvider:
        return provider

    try:
        with pytest.raises(type(operation_control)) as raised:
            await execute_expected_document_diagnostic(
                _input(RetrievalMode.BM25),
                provider_factory=factory,
                query_embedder=None,
            )
    finally:
        while owner.cancelling():
            owner.uncancel()

    assert provider.close_calls == 1
    assert raised.value is not operation_control
    assert _MARKER not in str(raised.value)
    assert operation_control.__traceback__ is None
    assert operation_control.__cause__ is None
    assert operation_control.__context__ is None


@pytest.mark.asyncio
async def test_blocked_thread_embedding_drains_after_repeated_cancellation() -> None:
    started = threading.Event()
    release = threading.Event()

    class ThreadEmbedder(FakeEmbedder):
        async def embed_query(self, query_text: str) -> QueryEmbedding:
            def blocked() -> QueryEmbedding:
                assert query_text == _MARKER
                started.set()
                assert release.wait(timeout=5)
                return QueryEmbedding(vector=(0.25, -0.5), client_duration_ms=1.0)

            return await asyncio.to_thread(blocked)

    factory_calls = 0

    async def factory() -> FakeProvider:
        nonlocal factory_calls
        factory_calls += 1
        return FakeProvider()

    task = asyncio.create_task(
        execute_expected_document_diagnostic(
            _input(RetrievalMode.VECTOR),
            provider_factory=factory,
            query_embedder=ThreadEmbedder(),
        )
    )
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert factory_calls == 0
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "embedding",
    [
        QueryEmbedding(vector=(10**10_000, 0.0), client_duration_ms=1.0),
        QueryEmbedding(vector=(0.0, 1.0), client_duration_ms=10**10_000),
    ],
)
async def test_huge_embedding_numbers_fail_fixed_before_provider(embedding: QueryEmbedding) -> None:
    factory_calls = 0

    async def factory() -> FakeProvider:
        nonlocal factory_calls
        factory_calls += 1
        return FakeProvider()

    with pytest.raises(DiagnosticRetrievalFailure):
        await execute_expected_document_diagnostic(
            _input(RetrievalMode.VECTOR),
            provider_factory=factory,
            query_embedder=FakeEmbedder(embedding),
        )
    assert factory_calls == 0


@pytest.mark.asyncio
async def test_embedder_call_failure_is_fixed_and_retains_no_sensitive_context() -> None:
    class HostileEmbedder(FakeEmbedder):
        def embed_query(self, query_text: str) -> object:  # type: ignore[override]
            raise RuntimeError(f"{query_text}-call-failure")

    factory_calls = 0

    async def factory() -> FakeProvider:
        nonlocal factory_calls
        factory_calls += 1
        return FakeProvider()

    with pytest.raises(DiagnosticRetrievalFailure) as raised:
        await execute_expected_document_diagnostic(
            _input(RetrievalMode.VECTOR),
            provider_factory=factory,
            query_embedder=HostileEmbedder(),
        )

    assert factory_calls == 0
    assert _MARKER not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [RetrievalMode.VECTOR, RetrievalMode.BM25])
async def test_child_task_start_failure_is_fixed_and_disposes_unstarted_coroutine(
    monkeypatch: pytest.MonkeyPatch,
    mode: RetrievalMode,
) -> None:
    marker = "task-start-secret-marker"
    provider = FakeProvider()
    factory_calls = 0

    async def factory() -> FakeProvider:
        nonlocal factory_calls
        factory_calls += 1
        return provider

    def fail_task_start(_: object) -> object:
        raise RuntimeError(marker)

    monkeypatch.setattr(diagnostic_module.asyncio, "create_task", fail_task_start)
    with pytest.raises(DiagnosticRetrievalFailure) as raised:
        await execute_expected_document_diagnostic(
            _input(mode),
            provider_factory=factory,
            query_embedder=FakeEmbedder() if mode is RetrievalMode.VECTOR else None,
        )

    assert _MARKER not in str(raised.value)
    assert marker not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    if mode is RetrievalMode.VECTOR:
        assert factory_calls == 0
    else:
        assert factory_calls == 1
        assert provider.query_calls == 1
        assert provider.close_calls == 1


def test_sensitive_internal_results_have_value_free_reprs() -> None:
    marker = "filter-value-marker"
    attribute = DiagnosticAttributeValue(
        field="category",
        state=DiagnosticAttributeState.PRESENT_VALUE,
        value=marker,
    )
    target = DiagnosticTargetObservation(
        target_document_id=_TARGET,
        available=True,
        bm25_score=_score(ScoreKind.BM25, 1.0, direct=True),
        vector_distance=None,
        attributes=(attribute,),
    )
    row = DiagnosticCandidateRow(
        document_id=_OTHER,
        rank=1,
        score=_score(ScoreKind.BM25, 1.0, direct=False),
    )
    candidate = DiagnosticCandidateList(
        role=DiagnosticSubqueryRole.STORED_QUERY_BM25_CANDIDATES,
        requested_limit=50,
        rows=(row,),
    )
    result = DiagnosticProviderResult(
        namespace="m5_fixture",
        target=target,
        candidate_lists=(candidate,),
        client_duration_ms=1.0,
    )
    for value in (attribute, target, row, candidate, result):
        rendered = repr(value)
        assert marker not in rendered
        assert str(_OTHER) not in rendered
