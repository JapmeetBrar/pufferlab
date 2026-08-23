from datetime import UTC, datetime
from uuid import UUID

import pytest
from pufferlab.application.evaluation_forensics import (
    analyze_counterfactual_probe,
    annotate_primary_with_exact_grades,
    build_forensic_observations,
    exact_qrel_grades,
    failed_counterfactual_probe,
)
from pufferlab.contracts.common import (
    ObservedScore,
    ScoreDirection,
    ScoreKind,
    ScoreSource,
)
from pufferlab.contracts.evals import Qrel
from pufferlab.contracts.forensics import (
    EvidenceCertainty,
    EvidenceOrigin,
    ForensicCode,
    ForensicWarningCode,
    RrfContributionEvidenceValue,
)
from pufferlab.contracts.retrieval import (
    LexicalSpec,
    RerankerSpec,
    RetrievalConfig,
    RetrievalConfigSummary,
    RetrievalMode,
    RrfSpec,
    VectorSpec,
)
from pufferlab.contracts.search import (
    ConfigSearchResult,
    RetrievalStage,
    SearchCompareResponse,
    SearchHit,
    StageMembership,
)
from pufferlab.retrieval.types import (
    HybridProbeCandidate,
    HybridProbeExecuteResult,
    HybridProbeStageMembership,
)

_NOW = datetime(2026, 8, 23, 12, tzinfo=UTC)
_DATASET_ID = UUID(int=10)
_QUERY_ID = UUID(int=11)


def _score(kind: ScoreKind, value: float) -> ObservedScore:
    return ObservedScore(
        kind=kind,
        value=value,
        direction=(
            ScoreDirection.LOWER_IS_BETTER
            if kind is ScoreKind.VECTOR_DISTANCE
            else ScoreDirection.HIGHER_IS_BETTER
        ),
        source=(
            ScoreSource.RERANKER if kind is ScoreKind.RERANKER else ScoreSource.TURBOPUFFER_DIST
        ),
    )


def _config(mode: RetrievalMode, *, value: int) -> RetrievalConfig:
    lexical = LexicalSpec() if mode is not RetrievalMode.VECTOR else None
    vector = VectorSpec(embedding_model="test-model") if mode is not RetrievalMode.BM25 else None
    rrf = RrfSpec(rank_constant=10, weights=(2.0, 1.0)) if "hybrid" in mode.value else None
    reranker = (
        RerankerSpec(provider="sentence_transformers", model="test", revision="v1", depth=50)
        if mode is RetrievalMode.HYBRID_RERANK
        else None
    )
    return RetrievalConfig(
        id=UUID(int=value),
        revision=1,
        name=mode.value,
        dataset_version_id=_DATASET_ID,
        mode=mode,
        result_k=50,
        candidate_k=100,
        consistency="strong",
        lexical=lexical,
        vector=vector,
        rrf=rrf,
        reranker=reranker,
        config_hash=f"hash-{value}",
        created_at=_NOW,
    )


def _summary(config: RetrievalConfig) -> RetrievalConfigSummary:
    return RetrievalConfigSummary(
        id=config.id,
        revision=config.revision,
        name=config.name,
        mode=config.mode,
        config_hash=config.config_hash,
    )


def _hit(document_id: UUID, *, rank: int, rrf_rank: int | None = None) -> SearchHit:
    memberships = []
    if rrf_rank is not None:
        memberships.append(
            StageMembership(
                stage=RetrievalStage.RRF,
                rank=rrf_rank,
                score=_score(ScoreKind.RRF, 1.0 / (10 + rrf_rank)),
            )
        )
    memberships.append(
        StageMembership(
            stage=RetrievalStage.FINAL,
            rank=rank,
            score=_score(ScoreKind.RRF, 1.0 / (10 + rank)),
        )
    )
    return SearchHit(
        document_id=document_id,
        external_id=f"doc-{document_id.int}",
        title="Authored test result",
        body_excerpt="Bounded authored excerpt.",
        final_rank=rank,
        final_score=_score(ScoreKind.RRF, 1.0 / (10 + rank)),
        stage_membership=memberships,
    )


def _primary(*results: ConfigSearchResult) -> SearchCompareResponse:
    return SearchCompareResponse(
        query_text="authored test query",
        query_id=_QUERY_ID,
        results=list(results),
        rank_movements=[],
        overlap=[],
        observability_notice="Only returned evidence is observable.",
    )


def test_exact_qrel_grades_replace_binary_labels_without_mutating_primary() -> None:
    config = _config(RetrievalMode.BM25, value=20)
    document_ids = [UUID(int=value) for value in range(100, 104)]
    result = ConfigSearchResult(
        config=_summary(config),
        hits=[_hit(document_id, rank=index) for index, document_id in enumerate(document_ids, 1)],
        timings=[],
        candidate_counts={"bm25_candidates": 4},
        warnings=[],
        trace_id=UUID(int=30),
    )
    primary = _primary(result)
    grades = exact_qrel_grades(
        [
            Qrel(document_id=document_ids[0], relevance_grade=0),
            Qrel(document_id=document_ids[1], relevance_grade=1),
            Qrel(document_id=document_ids[2], relevance_grade=2),
        ]
    )

    annotated = annotate_primary_with_exact_grades(primary, grades)

    assert [hit.relevance_grade for hit in annotated.results[0].hits] == [0, 1, 2, None]
    assert [hit.relevance_grade for hit in primary.results[0].hits] == [None] * 4
    with pytest.raises(ValueError, match="unique document"):
        exact_qrel_grades(
            [
                Qrel(document_id=document_ids[0], relevance_grade=1),
                Qrel(document_id=document_ids[0], relevance_grade=2),
            ]
        )


def test_weighted_rrf_probe_is_counterfactual_bounded_and_mismatch_warned() -> None:
    config = _config(RetrievalMode.HYBRID_RRF, value=21)
    documents = [UUID(int=value) for value in range(200, 260)]
    candidates = tuple(
        HybridProbeCandidate(
            document_id=document_id,
            stage_membership=(
                HybridProbeStageMembership(
                    stage=RetrievalStage.BM25_CANDIDATES,
                    rank=rank,
                    score=_score(ScoreKind.BM25, 100.0 - rank),
                ),
                HybridProbeStageMembership(
                    stage=RetrievalStage.VECTOR_CANDIDATES,
                    rank=rank,
                    score=_score(ScoreKind.VECTOR_DISTANCE, rank / 100.0),
                ),
            ),
        )
        for rank, document_id in enumerate(documents, start=1)
    )
    execution = HybridProbeExecuteResult(
        config_id=config.id,
        query_id=_QUERY_ID,
        trace_id=UUID(int=31),
        duration_ms=2.5,
        bm25_candidate_count=60,
        vector_candidate_count=60,
        candidates=candidates,
    )
    primary_result = ConfigSearchResult(
        config=_summary(config),
        hits=[
            _hit(document_id, rank=rank, rrf_rank=rank)
            for rank, document_id in enumerate(documents[:50], 1)
        ],
        timings=[],
        candidate_counts={"rrf": 50},
        warnings=[],
        trace_id=UUID(int=32),
    )
    analysis = analyze_counterfactual_probe(
        execution,
        observed_at=_NOW,
        config=config,
        primary=primary_result,
    )
    other = _config(RetrievalMode.BM25, value=22)
    other_result = ConfigSearchResult(
        config=_summary(other),
        hits=[],
        timings=[],
        candidate_counts={"bm25_candidates": 0},
        warnings=[],
        trace_id=UUID(int=33),
    )
    response = _primary(primary_result, other_result)
    observations = build_forensic_observations(
        primary=response,
        primary_observed_at=_NOW,
        config_ids=(config.id, other.id),
        configs={config.id: config, other.id: other},
        target_document_ids=(documents[-1],),
        probe_analyses={config.id: analysis},
        failed_probes={},
    )

    fusion = observations[0]
    assert fusion.code is ForensicCode.OUTSIDE_FUSION_TOP_K
    assert fusion.certainty is EvidenceCertainty.COUNTERFACTUAL
    assert fusion.origin is EvidenceOrigin.CLIENT_COMPUTED
    contributions = [
        item.value
        for item in fusion.evidence
        if isinstance(item.value, RrfContributionEvidenceValue)
    ]
    assert [value.contribution for value in contributions] == pytest.approx(
        [2.0 / 70.0, 1.0 / 70.0]
    )
    assert analysis.probe.warnings == []

    reversed_primary = primary_result.model_copy(
        update={
            "hits": [
                _hit(documents[1], rank=1, rrf_rank=1),
                _hit(documents[0], rank=2, rrf_rank=2),
                *primary_result.hits[2:],
            ]
        }
    )
    mismatch = analyze_counterfactual_probe(
        execution,
        observed_at=_NOW,
        config=config,
        primary=reversed_primary,
    )
    assert mismatch.probe.warnings[0].code is ForensicWarningCode.PROVENANCE_SNAPSHOT_DIFFERS


def test_failed_nonhybrid_and_reranker_rules_never_invent_a_cause() -> None:
    hybrid = _config(RetrievalMode.HYBRID_RRF, value=40)
    reranker = _config(RetrievalMode.HYBRID_RERANK, value=41)
    document_id = UUID(int=400)
    hybrid_result = ConfigSearchResult(
        config=_summary(hybrid),
        hits=[],
        timings=[],
        candidate_counts={"rrf": 0},
        warnings=[],
        trace_id=UUID(int=42),
    )
    reranked_hit = _hit(document_id, rank=2, rrf_rank=1).model_copy(
        update={
            "final_score": _score(ScoreKind.RERANKER, 0.25),
            "stage_membership": [
                StageMembership(
                    stage=RetrievalStage.RRF,
                    rank=1,
                    score=_score(ScoreKind.RRF, 0.5),
                ),
                StageMembership(
                    stage=RetrievalStage.RERANKER,
                    rank=2,
                    score=_score(ScoreKind.RERANKER, 0.25),
                ),
                StageMembership(
                    stage=RetrievalStage.FINAL,
                    rank=2,
                    score=_score(ScoreKind.RERANKER, 0.25),
                ),
            ],
        }
    )
    reranker_result = ConfigSearchResult(
        config=_summary(reranker),
        hits=[reranked_hit],
        timings=[],
        candidate_counts={"rrf": 1, "reranker": 1},
        warnings=[],
        trace_id=UUID(int=43),
    )
    failed = failed_counterfactual_probe(
        config_id=hybrid.id,
        observed_at=_NOW,
        trace_id=UUID(int=44),
    )
    observations = build_forensic_observations(
        primary=_primary(hybrid_result, reranker_result),
        primary_observed_at=_NOW,
        config_ids=(hybrid.id, reranker.id),
        configs={hybrid.id: hybrid, reranker.id: reranker},
        target_document_ids=(document_id,),
        probe_analyses={},
        failed_probes={hybrid.id: failed},
    )

    failed_observation, reranked = observations
    assert failed_observation.code is ForensicCode.NOT_OBSERVABLE
    assert failed_observation.certainty is EvidenceCertainty.INSUFFICIENT
    assert reranked.code is ForensicCode.RERANKED_DOWN
    assert reranked.certainty is EvidenceCertainty.OBSERVED
    assert {item.value.kind for item in reranked.evidence} == {"rank", "score"}
    rendered = " ".join(item.statement.lower() for item in observations)
    for forbidden in (
        "cache was cold",
        "query plan",
        "filter ran before",
        "probe caused the primary",
        "reranker rationale",
    ):
        assert forbidden not in rendered
