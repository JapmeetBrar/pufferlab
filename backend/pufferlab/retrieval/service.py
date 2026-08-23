"""BM25-versus-vector comparison orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import combinations
from time import perf_counter
from uuid import UUID, uuid4

from pufferlab.contracts.common import (
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.errors import ApiWarning
from pufferlab.contracts.filters import FilterNode
from pufferlab.contracts.retrieval import RetrievalConfigSummary, RetrievalMode
from pufferlab.contracts.search import (
    ConfigSearchResult,
    PairwiseOverlap,
    RankMovement,
    RetrievalStage,
    SearchCompareRequest,
    SearchCompareResponse,
    SearchHit,
    StageMembership,
    StageTiming,
    TimingStage,
)
from pufferlab.datasets.schema import NamespaceWriteSpec
from pufferlab.providers.errors import ProviderError
from pufferlab.providers.rerankers import RerankCandidate, Reranker, RerankResult
from pufferlab.providers.types import (
    DistanceMetric,
    DocumentId,
    ProviderDocument,
    ProviderHybridProbeResult,
    ProviderQueryResult,
)
from pufferlab.retrieval.config import SearchConfigCatalog, SeededSearchConfig
from pufferlab.retrieval.errors import (
    SearchError,
    config_not_found,
    embedding_failed,
    invalid_provider_result,
    invalid_search,
    provider_failed,
)
from pufferlab.retrieval.filter_validation import FixtureFilterValidator
from pufferlab.retrieval.rrf import RrfEntry, reconstruct_rrf
from pufferlab.retrieval.types import QueryEmbedder, QueryEmbedding, RetrievalProvider

_INCLUDE_ATTRIBUTES = ("external_id", "title", "body", "source_url")
_OBSERVABILITY_NOTICE = (
    "Ranks, scores, and timings report only returned rows and measured client wall-clock time; "
    "they do not infer unexposed turbopuffer execution internals."
)


class SearchCompareService:
    def __init__(
        self,
        *,
        namespace: str,
        catalog: SearchConfigCatalog,
        write_spec: NamespaceWriteSpec,
        provider: RetrievalProvider,
        query_embedder: QueryEmbedder,
        reranker: Reranker | None = None,
        trace_id_factory: Callable[[], UUID] = uuid4,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not namespace:
            raise ValueError("namespace must not be empty")
        self._namespace = namespace
        self._catalog = catalog
        self._filter_validator = FixtureFilterValidator(write_spec)
        self._provider = provider
        self._query_embedder = query_embedder
        self._reranker = reranker
        self._trace_id_factory = trace_id_factory
        self._clock = clock

    def list_configs(self) -> tuple[RetrievalConfigSummary, ...]:
        return self._catalog.summaries()

    async def compare(self, request: SearchCompareRequest) -> SearchCompareResponse:
        configs = self._resolve_configs(request.config_ids)
        self._validate_filter(request.filter_override)
        trace_id = self._trace_id_factory()
        results = [
            await self._run_config(config, request=request, trace_id=trace_id) for config in configs
        ]
        return SearchCompareResponse(
            query_text=request.query_text,
            query_id=request.query_id,
            results=results,
            rank_movements=_rank_movements(results),
            overlap=_pairwise_overlap(results),
            observability_notice=_OBSERVABILITY_NOTICE,
        )

    def _validate_filter(self, filter_override: FilterNode | None) -> None:
        if filter_override is not None:
            self._filter_validator.validate(filter_override)

    async def close(self) -> None:
        await self._provider.close()

    def _resolve_configs(self, config_ids: Sequence[UUID]) -> tuple[SeededSearchConfig, ...]:
        if len(set(config_ids)) != len(config_ids):
            raise invalid_search("config_ids must be distinct")
        configs: list[SeededSearchConfig] = []
        for config_id in config_ids:
            config = self._catalog.get(config_id)
            if config is None:
                raise config_not_found()
            configs.append(config)
        return tuple(configs)

    async def _run_config(
        self,
        config: SeededSearchConfig,
        *,
        request: SearchCompareRequest,
        trace_id: UUID,
    ) -> ConfigSearchResult:
        if config.mode is RetrievalMode.BM25:
            result = await self._query_bm25(config, request)
            candidate_stage = RetrievalStage.BM25_CANDIDATES
            embedding: QueryEmbedding | None = None
        elif config.mode is RetrievalMode.VECTOR:
            embedding = await self._embed_query(config, request.query_text)
            result = await self._query_ann(config, request, embedding)
            candidate_stage = RetrievalStage.VECTOR_CANDIDATES
        else:
            return await self._run_hybrid_config(config, request=request, trace_id=trace_id)

        if not math.isfinite(result.client_duration_ms) or result.client_duration_ms < 0:
            raise invalid_provider_result()
        hits = self._map_hits(
            result.documents,
            stage=candidate_stage,
            expected_document_ids=set(request.expected_document_ids),
            debug_provenance=request.debug_provenance,
        )
        timings = []
        if embedding is not None:
            timings.append(
                StageTiming(stage=TimingStage.EMBED, duration_ms=embedding.client_duration_ms)
            )
        timings.append(
            StageTiming(stage=TimingStage.TURBOPUFFER, duration_ms=result.client_duration_ms)
        )
        warnings = []
        if not hits:
            warnings.append(ApiWarning(code="empty_result", message="The search returned no rows."))

        return ConfigSearchResult(
            config=config.summary,
            hits=hits,
            timings=timings,
            candidate_counts={candidate_stage.value: len(result.documents)},
            warnings=warnings,
            trace_id=trace_id,
        )

    async def _run_hybrid_config(
        self,
        config: SeededSearchConfig,
        *,
        request: SearchCompareRequest,
        trace_id: UUID,
    ) -> ConfigSearchResult:
        embedding = await self._embed_query(config, request.query_text)
        fused = await self._query_hybrid_rrf(config, request, embedding)
        _validate_duration(fused.client_duration_ms)
        _validate_documents(fused.documents, expected_score_kind=ScoreKind.RRF)

        warnings: list[ApiWarning] = []
        probe: ProviderHybridProbeResult | None = None
        reconstruction: tuple[RrfEntry, ...] = ()
        probe_duration_ms: float | None = None
        fusion_duration_ms: float | None = None
        if request.debug_provenance:
            try:
                probe = await self._probe_hybrid_candidates(config, request, embedding)
                _validate_duration(probe.client_duration_ms)
                _validate_documents(
                    probe.bm25_documents,
                    expected_score_kind=ScoreKind.BM25,
                )
                _validate_documents(
                    probe.ann_documents,
                    expected_score_kind=ScoreKind.VECTOR_DISTANCE,
                )
                probe_duration_ms = probe.client_duration_ms
                fusion_start = self._clock()
                reconstruction = reconstruct_rrf(
                    (
                        tuple(document.id for document in probe.bm25_documents),
                        tuple(document.id for document in probe.ann_documents),
                    ),
                    rank_constant=_required_rrf_rank_constant(config),
                    weights=_required_rrf_weights(config),
                )
                reconstructed_prefix = tuple(
                    entry.document_id for entry in reconstruction[: len(fused.documents)]
                )
                if reconstructed_prefix != tuple(document.id for document in fused.documents):
                    warnings.append(
                        ApiWarning(
                            code="provenance_snapshot_differs",
                            message=(
                                "The separate debug probe reconstruction differs from the "
                                "production RRF order; concurrent writes or score ties may apply."
                            ),
                        )
                    )
                fusion_duration_ms = max(0.0, (self._clock() - fusion_start) * 1000.0)
            except Exception:
                probe = None
                reconstruction = ()
                warnings.append(
                    ApiWarning(
                        code="provenance_probe_failed",
                        message=(
                            "Production results are available, but the optional raw-list "
                            "provenance probe failed."
                        ),
                    )
                )

        final_documents = fused.documents[: config.result_k]
        rerank_result: RerankResult | None = None
        if config.mode is RetrievalMode.HYBRID_RERANK and fused.documents:
            final_documents, rerank_result = await self._rerank(
                config,
                request.query_text,
                fused.documents,
            )

        hits = self._map_hybrid_hits(
            fused_documents=fused.documents,
            final_documents=final_documents,
            probe=probe,
            reconstruction=reconstruction,
            expected_document_ids=set(request.expected_document_ids),
            debug_provenance=request.debug_provenance,
            reranked=rerank_result is not None,
        )
        timings = [
            StageTiming(stage=TimingStage.EMBED, duration_ms=embedding.client_duration_ms),
            StageTiming(stage=TimingStage.TURBOPUFFER, duration_ms=fused.client_duration_ms),
        ]
        if probe_duration_ms is not None:
            timings.append(
                StageTiming(
                    stage=TimingStage.PROVENANCE_PROBE,
                    duration_ms=probe_duration_ms,
                )
            )
        if fusion_duration_ms is not None:
            timings.append(StageTiming(stage=TimingStage.FUSION, duration_ms=fusion_duration_ms))
        if rerank_result is not None:
            timings.append(
                StageTiming(
                    stage=TimingStage.RERANK,
                    duration_ms=rerank_result.client_duration_ms,
                )
            )

        candidate_counts = {RetrievalStage.RRF.value: len(fused.documents)}
        if probe is not None:
            candidate_counts.update(
                {
                    RetrievalStage.BM25_CANDIDATES.value: len(probe.bm25_documents),
                    RetrievalStage.VECTOR_CANDIDATES.value: len(probe.ann_documents),
                }
            )
        if rerank_result is not None:
            candidate_counts[RetrievalStage.RERANKER.value] = len(rerank_result.scores)
        if not hits:
            warnings.append(ApiWarning(code="empty_result", message="The search returned no rows."))

        return ConfigSearchResult(
            config=config.summary,
            hits=hits,
            timings=timings,
            candidate_counts=candidate_counts,
            warnings=warnings,
            trace_id=trace_id,
        )

    async def _query_bm25(
        self,
        config: SeededSearchConfig,
        request: SearchCompareRequest,
    ) -> ProviderQueryResult:
        if config.text_attribute is None:
            raise invalid_search("BM25 configuration is incomplete")
        failure: SearchError | None = None
        try:
            return await self._provider.query_bm25(
                namespace=self._namespace,
                text_attribute=config.text_attribute,
                query_text=request.query_text,
                top_k=config.result_k,
                include_attributes=_INCLUDE_ATTRIBUTES,
                filters=request.filter_override,
                consistency=config.consistency,
                vector_attributes=("vector",),
            )
        except ProviderError:
            raise
        except Exception:
            failure = provider_failed("query_bm25")
        assert failure is not None
        raise failure from None

    async def _query_ann(
        self,
        config: SeededSearchConfig,
        request: SearchCompareRequest,
        embedding: QueryEmbedding,
    ) -> ProviderQueryResult:
        if config.vector_attribute is None or config.distance_metric is None:
            raise invalid_search("vector configuration is incomplete")
        failure: SearchError | None = None
        try:
            return await self._provider.query_ann(
                namespace=self._namespace,
                vector_attribute=config.vector_attribute,
                query_vector=embedding.vector,
                top_k=config.result_k,
                include_attributes=_INCLUDE_ATTRIBUTES,
                filters=request.filter_override,
                consistency=config.consistency,
                distance_metric=config.distance_metric,
            )
        except ProviderError:
            raise
        except Exception:
            failure = provider_failed("query_ann")
        assert failure is not None
        raise failure from None

    async def _query_hybrid_rrf(
        self,
        config: SeededSearchConfig,
        request: SearchCompareRequest,
        embedding: QueryEmbedding,
    ) -> ProviderQueryResult:
        text_attribute, vector_attribute, distance_metric = _required_hybrid_attributes(config)
        result_k = (
            _required_reranker_depth(config)
            if config.mode is RetrievalMode.HYBRID_RERANK
            else config.result_k
        )
        failure: SearchError | None = None
        try:
            return await self._provider.query_hybrid_rrf(
                namespace=self._namespace,
                text_attribute=text_attribute,
                query_text=request.query_text,
                vector_attribute=vector_attribute,
                query_vector=embedding.vector,
                candidate_k=config.candidate_k,
                result_k=result_k,
                include_attributes=_INCLUDE_ATTRIBUTES,
                rank_constant=_required_rrf_rank_constant(config),
                weights=_required_rrf_weights(config),
                filters=request.filter_override,
                consistency=config.consistency,
                distance_metric=distance_metric,
            )
        except ProviderError:
            raise
        except Exception:
            failure = provider_failed("query_hybrid_rrf")
        assert failure is not None
        raise failure from None

    async def _probe_hybrid_candidates(
        self,
        config: SeededSearchConfig,
        request: SearchCompareRequest,
        embedding: QueryEmbedding,
    ) -> ProviderHybridProbeResult:
        text_attribute, vector_attribute, distance_metric = _required_hybrid_attributes(config)
        return await self._provider.probe_hybrid_candidates(
            namespace=self._namespace,
            text_attribute=text_attribute,
            query_text=request.query_text,
            vector_attribute=vector_attribute,
            query_vector=embedding.vector,
            candidate_k=config.candidate_k,
            include_attributes=_INCLUDE_ATTRIBUTES,
            filters=request.filter_override,
            consistency=config.consistency,
            distance_metric=distance_metric,
        )

    async def _rerank(
        self,
        config: SeededSearchConfig,
        query_text: str,
        fused_documents: Sequence[ProviderDocument],
    ) -> tuple[tuple[ProviderDocument, ...], RerankResult]:
        reranker = self._reranker
        if reranker is None:
            raise invalid_search("hybrid reranker is not configured")
        if config.reranker_model != reranker.model or config.reranker_revision != reranker.revision:
            raise invalid_search("hybrid configuration does not match the reranker")

        depth = _required_reranker_depth(config)
        try:
            candidates = tuple(
                RerankCandidate(
                    document_id=document.id,
                    title=_required_string(document, "title"),
                    body=_required_string(document, "body"),
                )
                for document in fused_documents[:depth]
            )
        except (TypeError, ValueError):
            raise invalid_provider_result() from None
        failure: SearchError | None = None
        result: RerankResult | None = None
        try:
            result = await reranker.rerank(query_text=query_text, candidates=candidates)
        except Exception:
            failure = provider_failed("rerank")
        if failure is not None:
            raise failure from None
        assert result is not None
        _validate_duration(result.client_duration_ms)

        scores: dict[DocumentId, float] = {}
        for item in result.scores:
            if item.document_id in scores or not math.isfinite(item.score):
                raise invalid_provider_result()
            scores[item.document_id] = item.score
        if set(scores) != {candidate.document_id for candidate in candidates}:
            raise invalid_provider_result()

        original_rank = {
            document.id: rank for rank, document in enumerate(fused_documents[:depth], start=1)
        }
        reranked = sorted(
            fused_documents[:depth],
            key=lambda document: (
                -scores[document.id],
                original_rank[document.id],
                str(document.id),
            ),
        )
        final_documents = tuple(
            ProviderDocument(
                id=document.id,
                attributes=document.attributes,
                score=ObservedScore(
                    kind=ScoreKind.RERANKER,
                    value=scores[document.id],
                    direction=ScoreDirection.HIGHER_IS_BETTER,
                    source=ScoreSource.RERANKER,
                ),
            )
            for document in reranked[: config.result_k]
        )
        return final_documents, result

    async def _embed_query(
        self,
        config: SeededSearchConfig,
        query_text: str,
    ) -> QueryEmbedding:
        if (
            config.embedding_model != self._query_embedder.model
            or config.embedding_revision != self._query_embedder.revision
            or config.embedding_dimensions != self._query_embedder.dimensions
        ):
            raise invalid_search("vector configuration does not match the query embedder")

        failure: SearchError | None = None
        embedding: QueryEmbedding | None = None
        try:
            embedding = await self._query_embedder.embed_query(query_text)
        except Exception:
            failure = embedding_failed()
        if failure is not None:
            raise failure from None
        assert embedding is not None
        if len(embedding.vector) != self._query_embedder.dimensions or any(
            not math.isfinite(value) for value in embedding.vector
        ):
            raise embedding_failed()
        if not math.isfinite(embedding.client_duration_ms) or embedding.client_duration_ms < 0:
            raise embedding_failed()
        return embedding

    @staticmethod
    def _map_hybrid_hits(
        *,
        fused_documents: Sequence[ProviderDocument],
        final_documents: Sequence[ProviderDocument],
        probe: ProviderHybridProbeResult | None,
        reconstruction: Sequence[RrfEntry],
        expected_document_ids: set[UUID],
        debug_provenance: bool,
        reranked: bool,
    ) -> list[SearchHit]:
        failure: SearchError | None = None
        hits: list[SearchHit] = []
        try:
            fused_by_id = _document_rank_index(fused_documents)
            bm25_by_id = _document_rank_index(probe.bm25_documents) if probe is not None else {}
            ann_by_id = _document_rank_index(probe.ann_documents) if probe is not None else {}
            reconstructed_by_id = {
                entry.document_id: (rank, entry)
                for rank, entry in enumerate(reconstruction, start=1)
            }
            if len(reconstructed_by_id) != len(reconstruction):
                raise ValueError("duplicate reconstructed RRF document")

            seen_final_ids: set[DocumentId] = set()
            for final_rank, document in enumerate(final_documents, start=1):
                if document.id in seen_final_ids:
                    raise ValueError("duplicate final document")
                seen_final_ids.add(document.id)
                document_id = UUID(str(document.id))
                fused_item = fused_by_id.get(document.id)
                if fused_item is None:
                    raise ValueError("final document was absent from fused candidates")
                fused_rank, fused_document = fused_item
                memberships: list[StageMembership] = []
                if debug_provenance and probe is not None:
                    bm25_item = bm25_by_id.get(document.id)
                    if bm25_item is not None:
                        memberships.append(
                            StageMembership(
                                stage=RetrievalStage.BM25_CANDIDATES,
                                rank=bm25_item[0],
                                score=bm25_item[1].score,
                            )
                        )
                    ann_item = ann_by_id.get(document.id)
                    if ann_item is not None:
                        memberships.append(
                            StageMembership(
                                stage=RetrievalStage.VECTOR_CANDIDATES,
                                rank=ann_item[0],
                                score=ann_item[1].score,
                            )
                        )
                reconstructed_item = reconstructed_by_id.get(document.id)
                reconstructed = reconstructed_item[1] if reconstructed_item is not None else None
                rrf_score = (
                    ObservedScore(
                        kind=ScoreKind.RRF,
                        value=reconstructed.score,
                        direction=ScoreDirection.HIGHER_IS_BETTER,
                        source=ScoreSource.CLIENT_COMPUTED,
                    )
                    if reconstructed is not None
                    else fused_document.score
                )
                memberships.append(
                    StageMembership(
                        stage=RetrievalStage.RRF,
                        rank=(
                            reconstructed_item[0] if reconstructed_item is not None else fused_rank
                        ),
                        score=rrf_score,
                    )
                )
                if reranked:
                    memberships.append(
                        StageMembership(
                            stage=RetrievalStage.RERANKER,
                            rank=final_rank,
                            score=document.score,
                        )
                    )
                memberships.append(
                    StageMembership(
                        stage=RetrievalStage.FINAL,
                        rank=final_rank,
                        score=document.score,
                    )
                )
                hits.append(
                    SearchHit(
                        document_id=document_id,
                        external_id=_required_string(document, "external_id"),
                        title=_required_string(document, "title"),
                        body_excerpt=_required_string(document, "body"),
                        url=_optional_string(document, "source_url"),
                        relevance_grade=(
                            int(document_id in expected_document_ids)
                            if expected_document_ids
                            else None
                        ),
                        final_rank=final_rank,
                        final_score=document.score,
                        stage_membership=memberships,
                    )
                )
        except (TypeError, ValueError):
            failure = invalid_provider_result()
        if failure is not None:
            raise failure from None
        return hits

    @staticmethod
    def _map_hits(
        documents: Sequence[ProviderDocument],
        *,
        stage: RetrievalStage,
        expected_document_ids: set[UUID],
        debug_provenance: bool,
    ) -> list[SearchHit]:
        failure: SearchError | None = None
        hits: list[SearchHit] = []
        seen_document_ids: set[UUID] = set()
        try:
            for rank, document in enumerate(documents, start=1):
                document_id = UUID(str(document.id))
                if document_id in seen_document_ids or not math.isfinite(document.score.value):
                    raise ValueError("invalid provider document")
                seen_document_ids.add(document_id)
                external_id = _required_string(document, "external_id")
                title = _required_string(document, "title")
                body = _required_string(document, "body")
                source_url = _optional_string(document, "source_url")
                membership = []
                if debug_provenance:
                    membership.append(StageMembership(stage=stage, rank=rank, score=document.score))
                membership.append(
                    StageMembership(stage=RetrievalStage.FINAL, rank=rank, score=document.score)
                )
                hits.append(
                    SearchHit(
                        document_id=document_id,
                        external_id=external_id,
                        title=title,
                        body_excerpt=body,
                        url=source_url,
                        relevance_grade=(
                            int(document_id in expected_document_ids)
                            if expected_document_ids
                            else None
                        ),
                        final_rank=rank,
                        final_score=document.score,
                        stage_membership=membership,
                    )
                )
        except (TypeError, ValueError):
            failure = invalid_provider_result()
        if failure is not None:
            raise failure from None
        return hits


def _required_string(document: ProviderDocument, attribute: str) -> str:
    value = document.attributes.get(attribute)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing {attribute}")
    return value


def _optional_string(document: ProviderDocument, attribute: str) -> str | None:
    value = document.attributes.get(attribute)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"invalid {attribute}")
    return value


def _validate_duration(duration_ms: float) -> None:
    if not math.isfinite(duration_ms) or duration_ms < 0:
        raise invalid_provider_result()


def _validate_documents(
    documents: Sequence[ProviderDocument],
    *,
    expected_score_kind: ScoreKind,
) -> None:
    try:
        _document_rank_index(documents)
        if any(document.score.kind is not expected_score_kind for document in documents):
            raise ValueError("unexpected provider score kind")
    except (TypeError, ValueError):
        raise invalid_provider_result() from None


def _document_rank_index(
    documents: Sequence[ProviderDocument],
) -> dict[DocumentId, tuple[int, ProviderDocument]]:
    indexed: dict[DocumentId, tuple[int, ProviderDocument]] = {}
    for rank, document in enumerate(documents, start=1):
        if document.id in indexed or not math.isfinite(document.score.value):
            raise ValueError("invalid provider documents")
        indexed[document.id] = (rank, document)
    return indexed


def _required_hybrid_attributes(
    config: SeededSearchConfig,
) -> tuple[str, str, DistanceMetric]:
    if (
        config.text_attribute is None
        or config.vector_attribute is None
        or config.distance_metric is None
    ):
        raise invalid_search("hybrid configuration is incomplete")
    return config.text_attribute, config.vector_attribute, config.distance_metric


def _required_rrf_rank_constant(config: SeededSearchConfig) -> int:
    if config.rrf_rank_constant is None:
        raise invalid_search("hybrid RRF configuration is incomplete")
    return config.rrf_rank_constant


def _required_rrf_weights(config: SeededSearchConfig) -> tuple[float, float]:
    if config.rrf_weights is None:
        raise invalid_search("hybrid RRF configuration is incomplete")
    return config.rrf_weights


def _required_reranker_depth(config: SeededSearchConfig) -> int:
    if config.reranker_depth is None:
        raise invalid_search("hybrid reranker configuration is incomplete")
    return config.reranker_depth


def _rank_movements(results: Sequence[ConfigSearchResult]) -> list[RankMovement]:
    all_document_ids = sorted(
        {hit.document_id for result in results for hit in result.hits},
        key=str,
    )
    movements = []
    for document_id in all_document_ids:
        ranks = {
            result.config.id: next(
                (hit.final_rank for hit in result.hits if hit.document_id == document_id),
                None,
            )
            for result in results
        }
        present_ranks = [rank for rank in ranks.values() if rank is not None]
        max_delta = (
            max(present_ranks) - min(present_ranks)
            if len(present_ranks) == len(results) and present_ranks
            else None
        )
        movements.append(
            RankMovement(
                document_id=document_id,
                ranks_by_config=ranks,
                max_absolute_delta=max_delta,
            )
        )
    return movements


def _pairwise_overlap(results: Sequence[ConfigSearchResult]) -> list[PairwiseOverlap]:
    overlaps = []
    for left, right in combinations(results, 2):
        left_ids = {hit.document_id for hit in left.hits}
        right_ids = {hit.document_id for hit in right.hits}
        intersection_count = len(left_ids & right_ids)
        union_count = len(left_ids | right_ids)
        overlaps.append(
            PairwiseOverlap(
                left_config_id=left.config.id,
                right_config_id=right.config.id,
                left_count=len(left_ids),
                right_count=len(right_ids),
                intersection_count=intersection_count,
                jaccard=intersection_count / union_count if union_count else 1.0,
            )
        )
    return overlaps
