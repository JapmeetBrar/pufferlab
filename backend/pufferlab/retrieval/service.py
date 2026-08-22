"""BM25-versus-vector comparison orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import combinations
from uuid import UUID, uuid4

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
from pufferlab.providers.types import ProviderDocument, ProviderQueryResult
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
        trace_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if not namespace:
            raise ValueError("namespace must not be empty")
        self._namespace = namespace
        self._catalog = catalog
        self._filter_validator = FixtureFilterValidator(write_spec)
        self._provider = provider
        self._query_embedder = query_embedder
        self._trace_id_factory = trace_id_factory

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
        embedding: QueryEmbedding | None = None
        if config.mode is RetrievalMode.BM25:
            result = await self._query_bm25(config, request)
            candidate_stage = RetrievalStage.BM25_CANDIDATES
        else:
            embedding = await self._embed_query(config, request.query_text)
            result = await self._query_ann(config, request, embedding)
            candidate_stage = RetrievalStage.VECTOR_CANDIDATES

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
