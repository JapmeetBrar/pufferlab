import { useMutation, useQuery } from "@tanstack/react-query";
import { type FormEvent, useEffect, useRef, useState } from "react";

import {
  ApiRequestError,
  compareSearchConfigs,
  getCapabilities,
  getRetrievalConfigs,
  type CapabilitiesResponse,
  type RetrievalConfigListResponse,
  type SearchCompareRequest,
} from "../../api/client";
import { configLabel } from "../configLabels";
import { ComparisonResults } from "./ComparisonResults";
import { currentCapabilityReadiness } from "./capabilityState";

type Config = RetrievalConfigListResponse["configs"][number];
type LivePlaygroundCapability = CapabilitiesResponse["live_playground"];
type CapabilityAction = NonNullable<LivePlaygroundCapability["next_action"]>;

const requirementLabels: Record<LivePlaygroundCapability["requirements"][number], string> = {
  api_key: "Server API key",
  search_namespace: "Search namespace",
  region: "Provider region",
  live_search_runtime: "Local live-search runtime",
  owned_tiny_receipt_invalid: "Valid owned-tiny receipt",
  owned_tiny_credential_mismatch: "Creating credential",
  owned_tiny_region_mismatch: "Creating region",
};

const actionGuidance: Record<CapabilityAction, { heading: string; instruction: string; command: string }> = {
  configure_api_key: {
    heading: "Configure the server API key",
    instruction: "Set the turbopuffer API key only in the server environment, then restart PufferLab.",
    command: "uv run pufferlab doctor --mode live-tiny",
  },
  configure_search_namespace: {
    heading: "Create or select the owned tiny namespace",
    instruction:
      "Run the generated tiny ingestion, copy the authenticated show-tiny assignment into the server environment, then restart PufferLab.",
    command: "uv run pufferlab dataset ingest-tiny",
  },
  configure_region: {
    heading: "Configure the provider region",
    instruction: "Set TURBOPUFFER_REGION in the server environment, then restart PufferLab.",
    command: "uv run pufferlab doctor --mode live-tiny",
  },
  install_live_search_runtime: {
    heading: "Install the local live-search runtime",
    instruction: "Install the locked optional model dependencies, then restart PufferLab.",
    command: "uv sync --locked --extra live-search",
  },
  resolve_owned_tiny_receipt: {
    heading: "Resolve the owned-tiny receipt",
    instruction:
      "The fixed local ownership receipt is invalid. Stop before live comparison and follow the receipt recovery guidance.",
    command: "uv run pufferlab doctor --mode live-tiny",
  },
  use_owned_tiny_credential: {
    heading: "Use the namespace's creating credential",
    instruction:
      "Configure the exact API credential that created the authenticated owned-tiny receipt, then restart PufferLab.",
    command: "uv run pufferlab doctor --mode live-tiny",
  },
  use_owned_tiny_region: {
    heading: "Use the namespace's creating region",
    instruction:
      "Use namespace show-tiny to recover the authenticated region assignment, update the server environment, then restart PufferLab.",
    command: "uv run pufferlab namespace show-tiny",
  },
};

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
  const capabilities = useQuery({
    queryKey: ["capabilities"],
    queryFn: ({ signal }) => getCapabilities(signal),
    retry: false,
  });
  const capabilityReadiness = currentCapabilityReadiness(capabilities);
  const livePlayground =
    capabilityReadiness.state === "action_required" ||
    capabilityReadiness.state === "locally_configured"
      ? capabilityReadiness.capability
      : undefined;
  const locallyConfigured = capabilityReadiness.state === "locally_configured";
  const configs = useQuery({
    queryKey: ["retrieval-configs"],
    queryFn: ({ signal }) => getRetrievalConfigs(signal),
    enabled: locallyConfigured,
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
    locallyConfigured &&
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
            Compare exact-token and semantic retrieval using observed evidence.
          </p>
        </div>
        <form className="query-console" onSubmit={submit}>
          {capabilityReadiness.state === "checking" && (
            <p className="capability-loading" role="status">
              Checking local live-search setup before enabling comparison…
            </p>
          )}
          {capabilityReadiness.state === "unavailable" && (
            <div className="capability-guidance capability-error" role="alert">
              <div>
                <strong>Local live-search setup could not be checked.</strong>
                <span>Comparison stays disabled until the provider-free capability check succeeds.</span>
              </div>
              <button type="button" onClick={() => void capabilities.refetch()}>
                Check again
              </button>
            </div>
          )}
          {livePlayground?.state === "action_required" && livePlayground.next_action !== null && (
            <section className="capability-guidance" aria-labelledby="capability-guidance-heading">
              <p className="eyebrow">Local setup required</p>
              <h2 id="capability-guidance-heading">
                {actionGuidance[livePlayground.next_action].heading}
              </h2>
              <p>{actionGuidance[livePlayground.next_action].instruction}</p>
              <code>{actionGuidance[livePlayground.next_action].command}</code>
              <p className="requirement-summary">
                Still needed: {livePlayground.requirements.map((requirement) => requirementLabels[requirement]).join(", ")}.
              </p>
            </section>
          )}
          {locallyConfigured && (
            <div className="capability-ready" role="note">
              <strong>Live search is locally configured.</strong>
              <span>
                Remote namespace health and authentication have not been checked. Comparing may load
                local models and contact turbopuffer.
              </span>
            </div>
          )}
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
          {locallyConfigured && configs.isPending && (
            <p className="connection-message" role="status">
              Loading configurations…
            </p>
          )}
          {locallyConfigured && configs.isError && (
            <div className="config-error" role="alert">
              <span>Configurations are unavailable.</span>
              <button type="button" onClick={() => void configs.refetch()}>
                Retry
              </button>
            </div>
          )}
          {locallyConfigured && configs.isSuccess && availableConfigs.length === 0 && (
            <div className="config-error" role="status">
              <span>No configurations have been seeded.</span>
              <button type="button" onClick={() => void configs.refetch()}>
                Check again
              </button>
            </div>
          )}
          <fieldset
            disabled={
              !locallyConfigured ||
              !configs.isSuccess ||
              availableConfigs.length < 2 ||
              comparison.isPending
            }
          >
            <legend>Compare configurations</legend>
            <div className="config-grid">
              <label>
                <span>Left</span>
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
                <span>Right</span>
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
            <p>Query and configs are saved in this URL.</p>
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
