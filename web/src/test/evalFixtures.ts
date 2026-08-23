import type {
  EvaluationConfigCatalogResponse,
  EvaluationDatasetListResponse,
  EvaluationQuerySetListResponse,
  EvaluationRunDetailResponse,
  EvaluationRunListResponse,
  EvaluationRunQueryDetailResponse,
  EvaluationRunQueryReplayResponse,
  RegressionResponse,
} from "../api/evaluations";

export const datasetId = "10000000-0000-4000-8000-000000000001";
export const querySetId = "20000000-0000-4000-8000-000000000002";
export const runId = "30000000-0000-4000-8000-000000000003";
export const baselineId = "40000000-0000-4000-8000-000000000004";
export const candidateIds = [
  "50000000-0000-4000-8000-000000000005",
  "60000000-0000-4000-8000-000000000006",
  "70000000-0000-4000-8000-000000000007",
] as const;
export const queryId = "80000000-0000-4000-8000-000000000008";
export const documentId = "90000000-0000-4000-8000-000000000009";
export const secondDocumentId = "91000000-0000-4000-8000-000000000009";
export const primaryObservedAt = "2026-08-23T13:00:00Z";
export const primaryTrace = "a1000000-0000-4000-8000-000000000001";
export const rightPrimaryTrace = "a2000000-0000-4000-8000-000000000002";
export const probeTrace = "a3000000-0000-4000-8000-000000000003";
export const failedProbeTrace = "a4000000-0000-4000-8000-000000000004";

type RunView = EvaluationRunListResponse["runs"][number];
type RunStatus = RunView["run"]["status"];
type Metric = RunView["run"]["summaries"][number]["metrics"][number];

export const evaluationConfigs: RunView["configs"] = [
  { id: baselineId, name: "BM25 baseline", mode: "bm25", revision: 1, config_hash: "bm25-hash" },
  { id: candidateIds[0], name: "ANN candidate", mode: "vector", revision: 1, config_hash: "ann-hash" },
  { id: candidateIds[1], name: "Server RRF", mode: "hybrid_rrf", revision: 1, config_hash: "rrf-hash" },
  { id: candidateIds[2], name: "Local reranker", mode: "hybrid_rerank", revision: 1, config_hash: "rerank-hash" },
];

function metrics(synthetic: boolean): Metric[] {
  return [
    { name: "ndcg@10", value: 0.4567, sample_count: 50 },
    { name: "recall@50", value: 0.72, sample_count: 50 },
    { name: "mrr@10", value: 0.5123, sample_count: 50 },
    { name: "latency_p50_ms", value: synthetic ? null : 121.4, sample_count: synthetic ? 0 : 50 },
    { name: "latency_p95_ms", value: synthetic ? null : 242.8, sample_count: synthetic ? 0 : 50 },
    { name: "error_rate", value: 0, sample_count: 50 },
  ];
}

export function makeRunView(
  status: RunStatus = "completed",
  options: { id?: string; synthetic?: boolean; completedQueries?: number; withSummaries?: boolean } = {},
): RunView {
  const synthetic = options.synthetic ?? false;
  const completedQueries = options.completedQueries ?? (status === "completed" ? 50 : 12);
  const withSummaries = options.withSummaries ?? status === "completed";
  return {
    dataset_version_id: datasetId,
    data_origin: synthetic ? "synthetic_demo" : "live",
    configs: evaluationConfigs,
    completed_attempts: completedQueries * 4,
    total_attempts: 200,
    original_stage_evidence_available: false,
    live_replay_policy_permitted: !synthetic,
    run: {
      contract_version: 1,
      id: options.id ?? runId,
      status,
      query_set: {
        id: querySetId,
        name: "Canonical 50-query suite",
        version: "1",
        content_hash: "query-hash",
        query_count: 50,
      },
      baseline_config_id: baselineId,
      candidate_config_ids: [...candidateIds],
      summaries: withSummaries
        ? evaluationConfigs.map((config) => ({
            config_id: config.id,
            completed_queries: completedQueries,
            failed_queries: 0,
            metrics: metrics(synthetic),
          }))
        : [],
      completed_queries: completedQueries,
      total_queries: 50,
      random_seed: 20260822,
      environment: {
        max_concurrency: 4,
        platform: "test-platform",
        pufferlab_git_revision: "test-revision",
        python_version: "3.13",
        query_embedding_cache_enabled: true,
        timing_source: synthetic ? "synthetic_unavailable" : "perf_counter",
        turbopuffer_region: synthetic ? "unavailable" : "gcp-us-west1",
        warmup_query_count: synthetic ? 0 : 5,
      },
      created_at: "2026-08-23T12:00:00Z",
      started_at: status === "queued" ? null : "2026-08-23T12:01:00Z",
      completed_at: ["completed", "failed", "cancelled", "interrupted"].includes(status)
        ? "2026-08-23T12:06:00Z"
        : null,
      error: status === "failed"
        ? {
            code: "provider_error",
            message: "A safe provider failure was recorded.",
            retryable: true,
            trace_id: "a0000000-0000-4000-8000-00000000000a",
          }
        : null,
    },
  };
}

export function runDetail(view = makeRunView()): EvaluationRunDetailResponse {
  return { contract_version: 1, result: view };
}

export const syntheticDatasets: EvaluationDatasetListResponse = {
  contract_version: 1,
  datasets: [
    {
      data_origin: "synthetic_demo",
      dataset: {
        id: datasetId,
        slug: "pufferlab-authored-demo",
        version: "1",
        corpus_hash: "corpus-hash",
        namespace: "synthetic-unavailable",
        document_count: 100,
        status: "ready",
        data_origin: "synthetic_demo",
        created_at: "2026-08-23T12:00:00Z",
        index_profile: {
          id: "test-index-profile",
          schema_hash: "schema-hash",
          distance_metric: "cosine_distance",
          embedding_model: "test-embedding-model",
          embedding_provider: "sentence_transformers",
          embedding_revision: "test-revision",
          vector_dimensions: 3,
          vector_attribute: "embedding",
          vector_dtype: "f32",
          fts_profile: {
            ascii_folding: false,
            b: 0.75,
            case_sensitive: false,
            k1: 1.2,
            k3: 8,
            language: "english",
            max_token_length: 39,
            remove_stopwords: false,
            stemming: true,
            tokenizer: "word_v4",
          },
        },
      },
    },
  ],
};

export const liveDatasets: EvaluationDatasetListResponse = {
  ...syntheticDatasets,
  datasets: syntheticDatasets.datasets.map((item) => ({
    ...item,
    data_origin: "live",
    dataset: { ...item.dataset, data_origin: "live", namespace: "live-test-namespace" },
  })),
};

export const querySets: EvaluationQuerySetListResponse = {
  contract_version: 1,
  dataset_version_id: datasetId,
  query_sets: [
    {
      data_origin: "synthetic_demo",
      query_set: {
        id: querySetId,
        dataset_version_id: datasetId,
        name: "Canonical 50-query suite",
        version: "1",
        content_hash: "query-hash",
        query_count: 50,
        created_at: "2026-08-23T12:00:00Z",
      },
    },
  ],
};

export const liveQuerySets: EvaluationQuerySetListResponse = {
  ...querySets,
  query_sets: querySets.query_sets.map((item) => ({ ...item, data_origin: "live" })),
};

export const configCatalog: EvaluationConfigCatalogResponse = {
  contract_version: 1,
  dataset_version_id: datasetId,
  data_origin: "synthetic_demo",
  configs: evaluationConfigs,
};

export const liveConfigCatalog: EvaluationConfigCatalogResponse = {
  ...configCatalog,
  data_origin: "live",
};

export const regressionResponse: RegressionResponse = {
  contract_version: 1,
  run_id: runId,
  data_origin: "synthetic_demo",
  baseline_config_id: baselineId,
  candidate_config_id: candidateIds[0],
  order: "regressions",
  limit: 10,
  coverage: {
    total_queries: 50,
    paired_queries: 44,
    excluded: [
      { status: "baseline_missing", count: 1 },
      { status: "candidate_missing", count: 1 },
      { status: "baseline_failed", count: 1 },
      { status: "candidate_failed", count: 1 },
      { status: "both_failed", count: 1 },
      { status: "no_positive_qrels", count: 1 },
    ],
  },
  rows: [
    {
      query_id: queryId,
      query_text: "authored sample query",
      baseline_config_id: baselineId,
      candidate_config_id: candidateIds[0],
      baseline_ndcg_at_10: 0.8,
      candidate_ndcg_at_10: 0.4,
      ndcg_delta: -0.4,
      recall_delta: -0.2,
      mrr_delta: -0.25,
      baseline_latency_ms: null,
      candidate_latency_ms: null,
      relevant_rank_changes: [
        { document_id: documentId, relevance_grade: 2, baseline_rank: 1, candidate_rank: 8 },
      ],
      playground_url: `/playground?run=${runId}&query=${queryId}&left=${baselineId}&right=${candidateIds[0]}&document=${documentId}`,
    },
  ],
};

function successOutcome(configId: string, rank: number): EvaluationRunQueryDetailResponse["outcomes"][number] {
  return {
    run_id: runId,
    query_id: queryId,
    config_id: configId,
    created_at: "2026-08-23T12:05:00Z",
    outcome: {
      contract_version: 1,
      kind: "success",
      ranked_document_ids: rank === 1 ? [documentId, secondDocumentId] : [secondDocumentId, documentId],
      metrics: { ndcg_at_10: 0.75, recall_at_50: 1, mrr_at_10: 0.5 },
      timing_source: "perf_counter",
      total_client_wall_latency_ms: 12.5,
      stage_timings: [{ stage: "total", duration_ms: 12.5, measurement: "client_wall_clock" }],
      candidate_counts: { final: 2 },
      warnings: [],
      trace_id: configId === baselineId ? primaryTrace : rightPrimaryTrace,
    },
  };
}

export function queryDetail(
  origin: "live" | "synthetic_demo" = "live",
): EvaluationRunQueryDetailResponse {
  return {
    contract_version: 1,
    run_id: runId,
    data_origin: origin,
    query: {
      id: queryId,
      external_id: "authored-query-8",
      text: "authored local query text",
      qrels: [
        { document_id: documentId, relevance_grade: 2 },
        { document_id: secondDocumentId, relevance_grade: 1 },
      ],
      tags: ["authored"],
      filters: null,
    },
    baseline_config_id: baselineId,
    candidate_config_ids: [...candidateIds],
    configs: evaluationConfigs,
    outcomes: origin === "synthetic_demo" ? [] : [
      successOutcome(baselineId, 1),
      successOutcome(candidateIds[0], 2),
      {
        run_id: runId,
        query_id: queryId,
        config_id: candidateIds[1],
        created_at: "2026-08-23T12:05:00Z",
        outcome: {
          contract_version: 1,
          kind: "failure",
          code: "provider_error",
          message: "A safe recorded failure.",
          operation: "evaluate_query",
          retryable: true,
          total_client_wall_latency_ms: 9,
          trace_id: "a5000000-0000-4000-8000-000000000005",
        },
      },
    ],
    rank_changes: candidateIds.map((candidateConfigId, index) => ({
      candidate_config_id: candidateConfigId,
      changes: [{
        document_id: documentId,
        relevance_grade: 2,
        baseline_rank: 1,
        candidate_rank: index === 0 ? 2 : null,
      }],
    })),
    attribution: {
      source_name: "PufferLab authored test data",
      source_url: "https://example.com/source",
      license_name: "Test license",
      license_url: "https://example.com/license",
    },
    original_stage_evidence_available: false,
    live_replay_policy_permitted: origin === "live",
  };
}

function evaluationConfig(configId: string): (typeof evaluationConfigs)[number] {
  const config = evaluationConfigs.find((item) => item.id === configId);
  if (config === undefined) throw new Error(`Missing authored config fixture ${configId}`);
  return config;
}

export const replayResponse: EvaluationRunQueryReplayResponse = {
  contract_version: 1,
  run_id: runId,
  query_id: queryId,
  data_origin: "live",
  config_ids: [baselineId, candidateIds[0]],
  primary_origin: "live_replay_primary",
  primary_observed_at: primaryObservedAt,
  original_stage_evidence_available: false,
  observability_notice: "Primary and probe evidence are separate observations.",
  primary: {
    contract_version: 1,
    query_id: queryId,
    query_text: "authored local query text",
    observability_notice: "Only returned ranks, scores, and stages are displayed.",
    overlap: [{
      left_config_id: baselineId,
      right_config_id: candidateIds[0],
      intersection_count: 1,
      left_count: 1,
      right_count: 1,
      jaccard: 1,
    }],
    rank_movements: [{
      document_id: documentId,
      ranks_by_config: { [baselineId]: 1, [candidateIds[0]]: 2 },
      max_absolute_delta: 1,
    }],
    results: [
      {
        config: evaluationConfig(baselineId),
        trace_id: primaryTrace,
        candidate_counts: { final: 1 },
        timings: [{ stage: "total", duration_ms: 11, measurement: "client_wall_clock" }],
        warnings: [],
        hits: [{
          document_id: documentId,
          external_id: "authored-doc-9",
          title: "Authored relevant document",
          body_excerpt: "A locally authored excerpt for deterministic browser tests.",
          url: "https://example.com/document",
          final_rank: 1,
          relevance_grade: 2,
          final_score: {
            value: 8.5,
            kind: "bm25",
            direction: "higher_is_better",
            source: "turbopuffer_dist",
          },
          highlights: [],
          stage_membership: [{
            stage: "final",
            rank: 1,
            score: {
              value: 8.5,
              kind: "bm25",
              direction: "higher_is_better",
              source: "turbopuffer_dist",
            },
          }],
          attributes: {},
        }],
      },
      {
        config: evaluationConfig(candidateIds[0]),
        trace_id: rightPrimaryTrace,
        candidate_counts: { final: 1 },
        timings: [{ stage: "total", duration_ms: 14, measurement: "client_wall_clock" }],
        warnings: [],
        hits: [{
          document_id: documentId,
          external_id: "authored-doc-9",
          title: "Authored relevant document",
          body_excerpt: "A locally authored excerpt for deterministic browser tests.",
          url: null,
          final_rank: 2,
          relevance_grade: 2,
          final_score: {
            value: 0.12,
            kind: "vector_distance",
            direction: "lower_is_better",
            source: "turbopuffer_dist",
          },
          highlights: [],
          stage_membership: [],
          attributes: {},
        }],
      },
    ],
  },
  counterfactual_probes: [{
    origin: "live_replay_counterfactual_probe",
    config_id: candidateIds[0],
    observed_at: "2026-08-23T13:00:01Z",
    trace_id: probeTrace,
    duration_ms: 3.5,
    bm25_candidate_count: 20,
    vector_candidate_count: 20,
    candidates: [{
      document_id: documentId,
      stage_membership: [{
        stage: "vector_candidates",
        rank: 3,
        score: {
          value: 0.12,
          kind: "vector_distance",
          direction: "lower_is_better",
          source: "turbopuffer_dist",
        },
      }],
    }],
    warnings: [],
  }],
  failed_counterfactual_probes: [{
    origin: "live_replay_counterfactual_probe",
    config_id: baselineId,
    observed_at: "2026-08-23T13:00:02Z",
    trace_id: failedProbeTrace,
    warning: { code: "provenance_probe_failed", message: "The separate probe was unavailable." },
  }],
  observations: [
    {
      code: "outside_vector_candidates",
      statement: "The target was absent from the returned bounded vector candidates.",
      config_id: candidateIds[0],
      document_id: documentId,
      origin: "live_replay_counterfactual_probe",
      observed_at: "2026-08-23T13:00:01Z",
      trace_id: probeTrace,
      certainty: "counterfactual",
      evidence: [
        { label: "rank", value: { kind: "rank", stage: "vector_candidates", rank: 3 }, origin: "live_replay_counterfactual_probe", observed_at: "2026-08-23T13:00:01Z", trace_id: probeTrace },
        { label: "score", value: { kind: "score", stage: "vector_candidates", score: { value: 0.12, kind: "vector_distance", direction: "lower_is_better", source: "turbopuffer_dist" } }, origin: "live_replay_counterfactual_probe", observed_at: "2026-08-23T13:00:01Z", trace_id: probeTrace },
        { label: "count", value: { kind: "candidate_count", stage: "vector_candidates", count: 20 }, origin: "live_replay_counterfactual_probe", observed_at: "2026-08-23T13:00:01Z", trace_id: probeTrace },
        { label: "presence", value: { kind: "presence", stage: "vector_candidates", present: true }, origin: "live_replay_counterfactual_probe", observed_at: "2026-08-23T13:00:01Z", trace_id: probeTrace },
        { label: "filter", value: { kind: "filter_result", field: "source", matched: true }, origin: "live_replay_counterfactual_probe", observed_at: "2026-08-23T13:00:01Z", trace_id: probeTrace },
        { label: "warning", value: { kind: "warning", code: "provenance_snapshot_differs" }, origin: "live_replay_counterfactual_probe", observed_at: "2026-08-23T13:00:01Z", trace_id: probeTrace },
      ],
    },
    {
      code: "outside_fusion_top_k",
      statement: "The bounded returned inputs support this arithmetic only.",
      config_id: candidateIds[0],
      document_id: documentId,
      origin: "client_computed",
      observed_at: "2026-08-23T13:00:01Z",
      trace_id: probeTrace,
      certainty: "counterfactual",
      evidence: [{
        label: "rrf_contribution",
        value: {
          kind: "rrf_contribution",
          stage: "vector_candidates",
          rank: 3,
          weight: 1,
          rank_constant: 60,
          contribution: 1 / 63,
        },
        origin: "live_replay_counterfactual_probe",
        observed_at: "2026-08-23T13:00:01Z",
        trace_id: probeTrace,
      }],
    },
  ],
};
