import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo } from "react";

import { ApiRequestError } from "../../api/client";
import {
  getEvaluationRegressions,
  getEvaluationRun,
  type RegressionQuery,
} from "../../api/evaluations";
import { AppLink, RouteHeading } from "../../app/router";
import { navigate } from "../../app/routing";
import {
  type MetricAggregate,
  MetricValue,
  OriginBadge,
  PageIntro,
  RequestErrorPanel,
  StatusBadge,
} from "./components";
import { formatDate } from "./formatters";

export const ACTIVE_RUN_POLL_INTERVAL_MS = 2_000;

const metricOrder: MetricAggregate["name"][] = [
  "ndcg@10",
  "recall@50",
  "mrr@10",
  "latency_p50_ms",
  "latency_p95_ms",
  "error_rate",
];

const metricLabels: Record<MetricAggregate["name"], string> = {
  "ndcg@10": "NDCG@10",
  "recall@50": "Recall@50",
  "mrr@10": "MRR@10",
  latency_p50_ms: "p50 client wall",
  latency_p95_ms: "p95 client wall",
  error_rate: "Error rate",
};

const exclusionLabels = {
  paired: "Paired",
  baseline_missing: "Baseline missing",
  candidate_missing: "Candidate missing",
  baseline_failed: "Baseline failed",
  candidate_failed: "Candidate failed",
  both_failed: "Both failed",
  no_positive_qrels: "No positive judgments",
} as const;

function isActiveStatus(status: string | undefined): boolean {
  return status === "queued" || status === "running";
}

function resolveRegressionState(search: string, candidateIds: readonly string[]): {
  query: RegressionQuery | null;
  canonicalSearch: string;
} {
  const fallbackCandidate = candidateIds[0];
  if (fallbackCandidate === undefined) return { query: null, canonicalSearch: "" };
  const current = new URLSearchParams(search);
  const requestedCandidate = current.get("candidate");
  const candidate = requestedCandidate !== null && candidateIds.includes(requestedCandidate)
    ? requestedCandidate
    : fallbackCandidate;
  const orderValue = current.get("order");
  const order: RegressionQuery["order"] = orderValue === "gains" ? "gains" : "regressions";
  const parsedLimit = Number(current.get("limit"));
  const limit = Number.isInteger(parsedLimit) && parsedLimit >= 1 && parsedLimit <= 50
    ? parsedLimit
    : 10;
  const canonical = new URLSearchParams({ candidate, order, limit: String(limit) });
  return {
    query: { candidate_config_id: candidate, order, limit },
    canonicalSearch: `?${canonical.toString()}`,
  };
}

function MetricsTable({
  view,
}: {
  view: Awaited<ReturnType<typeof getEvaluationRun>>["result"];
}) {
  const complete = view.run.status === "completed";
  if (view.run.summaries.length === 0) {
    return (
      <div className="dashboard-empty compact" role="status">
        <h3>No aggregate metrics yet</h3>
        <p>Durable summaries appear after enough query outcomes have been recorded.</p>
      </div>
    );
  }
  const summaryByConfig = new Map(view.run.summaries.map((summary) => [summary.config_id, summary]));

  return (
    <>
      <div className="metric-context" role="note">
        <strong>{complete ? "Final durable metrics" : "Partial durable metrics · not final"}</strong>
        <span>
          Latency is observed PufferLab client-wall time, not provider service time or a benchmark.
        </span>
      </div>
      <div className="table-scroll" role="region" aria-label="Quality metrics table" tabIndex={0}>
        <table className="metrics-table">
          <caption className="visually-hidden">
            {complete ? "Final" : "Partial"} metrics by retrieval configuration
          </caption>
          <thead>
            <tr>
              <th scope="col">Configuration</th>
              {metricOrder.map((name) => <th scope="col" key={name}>{metricLabels[name]}</th>)}
            </tr>
          </thead>
          <tbody>
            {view.configs.map((config) => {
              const summary = summaryByConfig.get(config.id);
              return (
                <tr key={config.id}>
                  <th scope="row">
                    {config.name}
                    <span className="table-subcopy">{config.mode.replaceAll("_", " ")}</span>
                    {summary !== undefined && (
                      <span className="table-subcopy">
                        {summary.completed_queries} completed · {summary.failed_queries} failed
                      </span>
                    )}
                  </th>
                  {metricOrder.map((name) => {
                    const metric = summary?.metrics.find((item) => item.name === name);
                    return (
                      <td key={name}>
                        {metric === undefined ? "Not recorded" : <MetricValue metric={metric} />}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

function RegressionSection({
  runId,
  view,
  search,
}: {
  runId: string;
  view: Awaited<ReturnType<typeof getEvaluationRun>>["result"];
  search: string;
}) {
  const regressionState = useMemo(
    () => resolveRegressionState(search, view.run.candidate_config_ids),
    [search, view.run.candidate_config_ids],
  );
  const query = regressionState.query;
  useEffect(() => {
    if (query !== null && search !== regressionState.canonicalSearch) {
      navigate(`/runs/${encodeURIComponent(runId)}${regressionState.canonicalSearch}`, {
        replace: true,
      });
    }
  }, [query, regressionState.canonicalSearch, runId, search]);

  const regressions = useQuery({
    queryKey: ["evaluation-regressions", runId, query],
    queryFn: ({ signal }) => {
      if (query === null) throw new Error("A candidate configuration is required");
      return getEvaluationRegressions(runId, query, signal);
    },
    enabled: query !== null,
    retry: false,
  });
  const configById = new Map(view.configs.map((config) => [config.id, config]));

  function updateUrl(next: Partial<{ candidate: string; order: string; limit: number }>) {
    if (query === null) return;
    const parameters = new URLSearchParams({
      candidate: next.candidate ?? query.candidate_config_id,
      order: next.order ?? query.order ?? "regressions",
      limit: String(next.limit ?? query.limit ?? 10),
    });
    navigate(`/runs/${encodeURIComponent(runId)}?${parameters.toString()}`);
  }

  return (
    <section className="regression-panel" aria-labelledby="regression-heading">
      <div className="section-heading-row">
        <div>
          <p className="eyebrow">Paired query evidence</p>
          <h2 id="regression-heading">Regressions and gains</h2>
        </div>
      </div>
      {query === null ? (
        <div className="dashboard-empty compact" role="status">
          <h3>No candidate configurations</h3>
          <p>This run has no candidate pair to compare.</p>
        </div>
      ) : (
        <>
          <div className="regression-filters" aria-label="Regression table controls">
            <label>
              <span>Candidate</span>
              <select
                value={query.candidate_config_id}
                onChange={(event) => updateUrl({ candidate: event.target.value })}
              >
                {view.run.candidate_config_ids.map((configId) => (
                  <option key={configId} value={configId}>
                    {configById.get(configId)?.name ?? configId}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Order</span>
              <select
                value={query.order}
                onChange={(event) => updateUrl({ order: event.target.value })}
              >
                <option value="regressions">Largest regressions</option>
                <option value="gains">Largest gains</option>
              </select>
            </label>
            <label>
              <span>Rows</span>
              <input
                type="number"
                min={1}
                max={50}
                value={query.limit}
                onChange={(event) => {
                  const value = event.currentTarget.valueAsNumber;
                  if (Number.isInteger(value) && value >= 1 && value <= 50) {
                    updateUrl({ limit: value });
                  }
                }}
              />
            </label>
          </div>

          {regressions.isPending && <p className="route-loading" role="status">Loading paired regressions…</p>}
          {regressions.isError && (
            <RequestErrorPanel
              error={regressions.error}
              heading="Regression evidence is unavailable."
              onRetry={() => void regressions.refetch()}
            />
          )}
          {regressions.isSuccess && (
            <>
              <div className="coverage-summary" aria-label="Regression coverage">
                <div>
                  <strong>{regressions.data.coverage.paired_queries} / {regressions.data.coverage.total_queries}</strong>
                  <span>paired queries</span>
                </div>
                <dl>
                  {regressions.data.coverage.excluded.map((item) => (
                    <div key={item.status}>
                      <dt>{exclusionLabels[item.status]}</dt>
                      <dd>{item.count}</dd>
                    </div>
                  ))}
                </dl>
              </div>
              {regressions.data.rows.length === 0 ? (
                <div className="dashboard-empty compact" role="status">
                  <h3>No paired rows for this view</h3>
                  <p>Review the explicit exclusion counts above; missing evidence is not scored as zero.</p>
                </div>
              ) : (
                <div
                  className="table-scroll"
                  role="region"
                  aria-label="Per-query regression table"
                  tabIndex={0}
                >
                  <table className="regression-table">
                    <caption className="visually-hidden">
                      Per-query {regressions.data.order} for the selected candidate
                    </caption>
                    <thead>
                      <tr>
                        <th scope="col">Query</th>
                        <th scope="col">NDCG@10</th>
                        <th scope="col">Recall delta</th>
                        <th scope="col">MRR delta</th>
                        <th scope="col">Client-wall latency</th>
                        <th scope="col">Relevant movement</th>
                      </tr>
                    </thead>
                    <tbody>
                      {regressions.data.rows.map((row) => (
                        <tr key={row.query_id}>
                          <th scope="row">
                            <span>{row.query_text}</span>
                            <AppLink href={row.playground_url}>Inspect recorded query</AppLink>
                          </th>
                          <td>
                            {row.baseline_ndcg_at_10.toFixed(3)} → {row.candidate_ndcg_at_10.toFixed(3)}
                            <span className={row.ndcg_delta < 0 ? "delta-negative" : "delta-positive"}>
                              {row.ndcg_delta >= 0 ? "+" : ""}{row.ndcg_delta.toFixed(3)}
                            </span>
                          </td>
                          <td>{row.recall_delta >= 0 ? "+" : ""}{row.recall_delta.toFixed(3)}</td>
                          <td>{row.mrr_delta >= 0 ? "+" : ""}{row.mrr_delta.toFixed(3)}</td>
                          <td>
                            {row.baseline_latency_ms == null || row.candidate_latency_ms == null
                              ? "Unavailable"
                              : `${row.baseline_latency_ms.toFixed(1)} → ${row.candidate_latency_ms.toFixed(1)} ms`}
                          </td>
                          <td>{row.relevant_rank_changes.length} judged documents</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}

export function RunDetailPage({
  runId,
  routeKey,
  search,
}: {
  runId: string;
  routeKey: string;
  search: string;
}) {
  const run = useQuery({
    queryKey: ["evaluation-run", runId],
    queryFn: ({ signal }) => getEvaluationRun(runId, signal),
    retry: false,
    refetchInterval: (query) =>
      isActiveStatus(query.state.data?.result.run.status) ? ACTIVE_RUN_POLL_INTERVAL_MS : false,
    refetchIntervalInBackground: false,
  });
  const view = run.data?.result;
  const requestError = run.error instanceof ApiRequestError ? run.error : null;

  return (
    <section className="dashboard-page run-detail-page">
      <div className="page-heading compact-heading">
        <p className="eyebrow">Durable evaluation record</p>
        <RouteHeading routeKey={routeKey}>
          {requestError?.status === 404 ? "Run not found" : "Evaluation run"}
        </RouteHeading>
        <PageIntro>
          Recorded progress and metrics come from SQLite. Opening or refreshing this page never
          performs provider work.
        </PageIntro>
      </div>

      {run.isPending && <p className="route-loading" role="status">Loading evaluation run…</p>}
      {run.isError && requestError?.status === 404 && (
        <div className="dashboard-empty" role="alert">
          <h2>No run matches this URL</h2>
          <p>The durable run may not exist in this local database.</p>
          <AppLink href="/runs">Return to run history</AppLink>
        </div>
      )}
      {run.isError && requestError?.status !== 404 && (
        <RequestErrorPanel
          error={run.error}
          heading="The evaluation run is unavailable."
          onRetry={() => void run.refetch()}
        />
      )}

      {view !== undefined && (
        <>
          <section className="run-overview" aria-labelledby="run-overview-heading">
            <div className="run-title-row">
              <div>
                <p className="run-id">{view.run.id}</p>
                <h2 id="run-overview-heading">{view.run.query_set.name}</h2>
              </div>
              <div className="run-badges">
                <StatusBadge status={view.run.status} />
                <OriginBadge origin={view.data_origin} />
              </div>
            </div>
            <div className="progress-card">
              <label htmlFor="run-query-progress">
                <strong>{view.run.completed_queries} of {view.run.total_queries}</strong>
                <span>durable query groups</span>
              </label>
              <progress
                id="run-query-progress"
                max={view.run.total_queries}
                value={view.run.completed_queries}
              />
              <p aria-live="polite" role="status">
                {view.run.completed_queries} of {view.run.total_queries} query groups complete;
                {" "}{view.completed_attempts} of {view.total_attempts} attempts durable.
                {isActiveStatus(view.run.status) ? " Progress refreshes automatically." : " Polling stopped."}
              </p>
            </div>
            <dl className="run-metadata">
              <div><dt>Dataset revision</dt><dd>{view.dataset_version_id}</dd></div>
              <div><dt>Created</dt><dd>{formatDate(view.run.created_at)}</dd></div>
              <div><dt>Started</dt><dd>{formatDate(view.run.started_at)}</dd></div>
              <div><dt>Completed</dt><dd>{formatDate(view.run.completed_at)}</dd></div>
            </dl>
            {view.run.error !== null && (
              <div className="inline-warning" role="alert">
                <strong>{view.run.error.message}</strong>
                <span>Trace {view.run.error.trace_id}</span>
              </div>
            )}
            {view.data_origin === "synthetic_demo" ? (
              <p className="synthetic-notice" role="note">
                <strong>Synthetic demo · read-only.</strong> Quality is recomputed from authored
                judgments and ranks. Timing is unavailable because no live requests occurred;
                create and replay actions are disabled.
              </p>
            ) : (
              <p className="evidence-policy" role="note">
                {view.live_replay_policy_permitted
                  ? "Policy permits a separate explicit live replay. Provider and namespace readiness have not been checked."
                  : "This recorded run is read-only under the current replay policy."}
              </p>
            )}
          </section>

          <section className="metrics-panel" aria-labelledby="metrics-heading">
            <div className="section-heading-row">
              <div>
                <p className="eyebrow">Aggregate evidence</p>
                <h2 id="metrics-heading">Quality and client timing</h2>
              </div>
            </div>
            <MetricsTable view={view} />
          </section>

          <RegressionSection runId={runId} view={view} search={search} />
        </>
      )}
    </section>
  );
}
