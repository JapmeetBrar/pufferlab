import type { ReactNode } from "react";

import { ApiRequestError } from "../../api/client";
import type { EvaluationRunListResponse } from "../../api/evaluations";

export type EvaluationRunView = EvaluationRunListResponse["runs"][number];
export type EvaluationRunStatus = EvaluationRunView["run"]["status"];
export type MetricAggregate = EvaluationRunView["run"]["summaries"][number]["metrics"][number];

const statusLabels: Record<EvaluationRunStatus, string> = {
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
  interrupted: "Interrupted",
};

export function StatusBadge({ status }: { status: EvaluationRunStatus }) {
  return (
    <span className={`status-badge status-${status}`}>
      <span aria-hidden="true">{status === "completed" ? "✓" : status === "running" ? "●" : "○"}</span>
      {statusLabels[status]}
    </span>
  );
}

export function OriginBadge({ origin }: { origin: EvaluationRunView["data_origin"] }) {
  return (
    <span className={`origin-badge origin-${origin}`}>
      {origin === "synthetic_demo" ? "Synthetic demo" : "Live recorded run"}
    </span>
  );
}

export function RequestErrorPanel({
  error,
  heading,
  onRetry,
}: {
  error: Error;
  heading: string;
  onRetry: () => void;
}) {
  const requestError = error instanceof ApiRequestError ? error : null;
  return (
    <section className="dashboard-error" role="alert">
      <h2>{heading}</h2>
      <p>{requestError?.detail.message ?? "An unexpected browser error occurred."}</p>
      {requestError !== null && <p className="trace-id">Trace {requestError.detail.trace_id}</p>}
      <button type="button" onClick={onRetry}>
        Try again
      </button>
    </section>
  );
}

function formatMetricValue(metric: MetricAggregate): string {
  if (metric.value === null) return "Unavailable";
  if (metric.name === "latency_p50_ms" || metric.name === "latency_p95_ms") {
    return `${metric.value.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms`;
  }
  if (metric.name === "error_rate") {
    return `${(metric.value * 100).toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
  }
  return metric.value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

export function MetricValue({ metric }: { metric: MetricAggregate }) {
  return (
    <span className={metric.value === null ? "metric-unavailable" : undefined}>
      <strong>{formatMetricValue(metric)}</strong>
      <span className="sample-count">sample count {metric.sample_count}</span>
    </span>
  );
}

export function PageIntro({ children }: { children: ReactNode }) {
  return <p className="page-intro">{children}</p>;
}
