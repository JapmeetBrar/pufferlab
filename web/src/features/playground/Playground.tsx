import { useMutation, useQuery } from "@tanstack/react-query";
import { type FormEvent, useEffect, useRef, useState } from "react";

import {
  ApiRequestError,
  compareSearchConfigs,
  getRetrievalConfigs,
  type RetrievalConfigListResponse,
  type SearchCompareRequest,
} from "../../api/client";
import { ComparisonResults } from "./ComparisonResults";

type Config = RetrievalConfigListResponse["configs"][number];

function configLabel(config: Config): string {
  return `${config.name} · ${config.mode.replaceAll("_", " ")}`;
}

function selectedConfigId(
  requestedId: string,
  configs: readonly Config[],
  preferredMode: Config["mode"],
  excludedId = "",
): string {
  if (configs.some((config) => config.id === requestedId && config.id !== excludedId)) {
    return requestedId;
  }
  return (
    configs.find((config) => config.mode === preferredMode && config.id !== excludedId)?.id ??
    configs.find((config) => config.id !== excludedId)?.id ??
    ""
  );
}

function ErrorPanel({ error, onRetry }: { error: Error; onRetry: () => void }) {
  const detail = error instanceof ApiRequestError ? error.detail : null;
  return (
    <section className="error-panel" role="alert" aria-labelledby="error-heading">
      <p className="eyebrow">Request interrupted</p>
      <h2 id="error-heading">The comparison could not be completed.</h2>
      <p>{detail?.message ?? "An unexpected browser error occurred."}</p>
      {detail !== null && <p className="trace-id">Trace {detail.trace_id}</p>}
      <button type="button" onClick={onRetry}>
        Try again
      </button>
    </section>
  );
}

export function Playground() {
  const headingRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    headingRef.current?.focus();
  }, []);
  const [queryText, setQueryText] = useState(
    () => new URLSearchParams(window.location.search).get("q") ?? "",
  );
  const [leftConfigId, setLeftConfigId] = useState(
    () => new URLSearchParams(window.location.search).get("left") ?? "",
  );
  const [rightConfigId, setRightConfigId] = useState(
    () => new URLSearchParams(window.location.search).get("right") ?? "",
  );
  const configs = useQuery({
    queryKey: ["retrieval-configs"],
    queryFn: ({ signal }) => getRetrievalConfigs(signal),
    retry: false,
  });
  const comparison = useMutation({ mutationFn: compareSearchConfigs });
  const availableConfigs = configs.data?.configs ?? [];
  const resolvedLeftId = selectedConfigId(leftConfigId, availableConfigs, "bm25");
  const resolvedRightId = selectedConfigId(
    rightConfigId,
    availableConfigs,
    "vector",
    resolvedLeftId,
  );
  const canCompare =
    !comparison.isPending &&
    queryText.trim().length > 0 &&
    resolvedLeftId.length > 0 &&
    resolvedRightId.length > 0;

  function submit(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const cleanQuery = queryText.trim();
    if (!canCompare || cleanQuery.length === 0) {
      return;
    }
    const request: SearchCompareRequest = {
      contract_version: 1,
      query_text: cleanQuery,
      config_ids: [resolvedLeftId, resolvedRightId],
      debug_provenance: true,
    };
    const params = new URLSearchParams();
    params.set("q", cleanQuery);
    params.set("left", resolvedLeftId);
    params.set("right", resolvedRightId);
    window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
    comparison.mutate(request);
  }

  return (
    <>
      <section className="playground-hero" aria-labelledby="playground-heading">
        <div className="hero-copy">
          <p className="eyebrow">Live retrieval playground</p>
          <h1 id="playground-heading" ref={headingRef} tabIndex={-1}>
            One query. Two retrieval instincts.
          </h1>
          <p className="lede">
            Put exact-token matching next to semantic similarity and inspect only what the system
            actually observed.
          </p>
        </div>
        <form className="query-console" onSubmit={submit}>
          <div className="query-field">
            <label htmlFor="query-text">Search query</label>
            <textarea
              id="query-text"
              name="query"
              rows={3}
              value={queryText}
              onChange={(event) => setQueryText(event.target.value)}
              placeholder="Try: find files containing an exact permission mode"
              required
            />
          </div>
          {configs.isPending && (
            <p className="connection-message" role="status">
              Loading retrieval configurations…
            </p>
          )}
          {configs.isError && (
            <div className="config-error" role="alert">
              <span>Retrieval configurations are unavailable.</span>
              <button type="button" onClick={() => void configs.refetch()}>
                Retry
              </button>
            </div>
          )}
          {configs.isSuccess && availableConfigs.length === 0 && (
            <div className="config-error" role="status">
              <span>No retrieval configurations have been seeded yet.</span>
              <button type="button" onClick={() => void configs.refetch()}>
                Check again
              </button>
            </div>
          )}
          <fieldset
            disabled={!configs.isSuccess || availableConfigs.length < 2 || comparison.isPending}
          >
            <legend>Configurations to compare</legend>
            <div className="config-grid">
              <label>
                <span>Left result set</span>
                <select
                  value={resolvedLeftId}
                  onChange={(event) => setLeftConfigId(event.target.value)}
                >
                  {availableConfigs.map((config) => (
                    <option key={config.id} value={config.id}>
                      {configLabel(config)}
                    </option>
                  ))}
                </select>
              </label>
              <span className="versus" aria-hidden="true">
                vs
              </span>
              <label>
                <span>Right result set</span>
                <select
                  value={resolvedRightId}
                  onChange={(event) => setRightConfigId(event.target.value)}
                >
                  {availableConfigs.map((config) => (
                    <option key={config.id} value={config.id}>
                      {configLabel(config)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </fieldset>
          <div className="submit-row">
            <button className="compare-button" type="submit" disabled={!canCompare}>
              {comparison.isPending ? "Comparing…" : "Compare results"}
            </button>
            <p>Query and config choices are saved in the copyable page URL.</p>
          </div>
          <div className="visually-hidden" role="status" aria-live="polite">
            {comparison.isPending && "Comparison is loading."}
            {comparison.isSuccess && "Comparison results loaded."}
          </div>
        </form>
      </section>

      {comparison.isError && <ErrorPanel error={comparison.error} onRetry={() => submit()} />}
      {comparison.isSuccess && (
        <section className="comparison" aria-labelledby="comparison-heading">
          <div className="comparison-heading">
            <div>
              <p className="eyebrow">Comparison</p>
              <h2 id="comparison-heading">Results for “{comparison.data.query_text}”</h2>
            </div>
            <span>{comparison.data.results.length} configurations returned</span>
          </div>
          <ComparisonResults response={comparison.data} />
        </section>
      )}
    </>
  );
}
