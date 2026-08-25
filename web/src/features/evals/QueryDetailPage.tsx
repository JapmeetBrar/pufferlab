import { useMutation, useQuery } from "@tanstack/react-query";
import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { ApiRequestError } from "../../api/client";
import {
  getEvaluationRunQuery,
  replayEvaluationRunQuery,
  type EvaluationRunQueryDetailResponse,
  type EvaluationRunQueryReplayRequest,
  type EvaluationRunQueryReplayResponse,
} from "../../api/evaluations";
import { AppLink, RouteHeading } from "../../app/router";
import { navigate } from "../../app/routing";
import { configLabel } from "../configLabels";
import { ComparisonResults } from "../playground/ComparisonResults";
import { safeSourceUrl } from "../playground/safeUrl";
import { relevanceLabel } from "../relevance";
import { RequestErrorPanel } from "./components";
import { ExpectedDocumentDiagnostic } from "./ExpectedDocumentDiagnostic";
import { ForensicDrawer } from "./ForensicDrawer";
import { ForensicEvidence } from "./ForensicEvidence";
import { formatDate } from "./formatters";
import {
  forensicHref,
  type ForensicRouteKind,
  type ForensicSelection,
  resolveForensicSelection,
} from "./queryState";

type QueryDetail = EvaluationRunQueryDetailResponse;
type OutcomeRecord = QueryDetail["outcomes"][number];
type ReplayResult = EvaluationRunQueryReplayResponse;
type ReplayScore = ReplayResult["primary"]["results"][number]["hits"][number]["final_score"];

function outcomeFor(detail: QueryDetail, configId: string): OutcomeRecord | undefined {
  return detail.outcomes.find((record) => record.config_id === configId);
}

function finalRank(record: OutcomeRecord | undefined, documentId: string): number | null {
  if (record?.outcome.kind !== "success") return null;
  const index = record.outcome.ranked_document_ids.indexOf(documentId);
  return index === -1 ? null : index + 1;
}

function formatMetric(value: number | null | undefined): string {
  return value == null ? "Unavailable" : value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function configName(detail: QueryDetail, configId: string): string {
  const config = detail.configs.find((item) => item.id === configId);
  return config === undefined ? configId : configLabel(config);
}

function documentTitle(detail: QueryDetail, documentId: string): string | null {
  return detail.judged_documents.find((item) => item.document_id === documentId)?.title ?? null;
}

function StoredOutcomes({ detail }: { detail: QueryDetail }) {
  return (
    <div className="table-scroll" role="region" aria-label="Recorded outcomes table" tabIndex={0}>
      <table className="query-outcome-table">
        <caption className="visually-hidden">Durable outcomes for the recorded query</caption>
        <thead>
          <tr>
            <th scope="col">Configuration</th>
            <th scope="col">Outcome</th>
            <th scope="col">NDCG@10</th>
            <th scope="col">Recall@50</th>
            <th scope="col">MRR@10</th>
            <th scope="col">Client wall time</th>
          </tr>
        </thead>
        <tbody>
          {detail.configs.map((config) => {
            const record = outcomeFor(detail, config.id);
            return (
              <tr key={config.id}>
                <th scope="row">{configLabel(config)}</th>
                <td>
                  {record === undefined ? "Not recorded" : record.outcome.kind === "success" ? "Success" : "Failed"}
                  {record?.outcome.kind === "failure" && (
                    <span className="table-subcopy">{record.outcome.message}</span>
                  )}
                </td>
                <td>{record?.outcome.kind === "success" ? formatMetric(record.outcome.metrics.ndcg_at_10) : "Unavailable"}</td>
                <td>{record?.outcome.kind === "success" ? formatMetric(record.outcome.metrics.recall_at_50) : "Unavailable"}</td>
                <td>{record?.outcome.kind === "success" ? formatMetric(record.outcome.metrics.mrr_at_10) : "Unavailable"}</td>
                <td>
                  {record?.outcome.kind === "success" && record.outcome.total_client_wall_latency_ms !== null
                    ? `${record.outcome.total_client_wall_latency_ms.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms`
                    : "Unavailable"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function JudgedDocuments({
  detail,
  selection,
  onInspect,
}: {
  detail: QueryDetail;
  selection: ForensicSelection;
  onInspect: (documentId: string, trigger: HTMLButtonElement) => void;
}) {
  const left = outcomeFor(detail, selection.left);
  const right = outcomeFor(detail, selection.right);
  return (
    <div className="table-scroll" role="region" aria-label="Judged documents table" tabIndex={0}>
      <table className="judgment-table">
        <caption className="visually-hidden">Judged documents and durable final ranks</caption>
        <thead>
          <tr>
            <th scope="col">Judged document</th>
            <th scope="col">Relevance</th>
            <th scope="col">{configName(detail, selection.left)} rank</th>
            <th scope="col">{configName(detail, selection.right)} rank</th>
            <th scope="col">Evidence</th>
          </tr>
        </thead>
        <tbody>
          {detail.query.qrels.map((qrel) => (
            <tr key={qrel.document_id}>
              <th scope="row" className="document-title-cell">
                {documentTitle(detail, qrel.document_id) ?? "Title unavailable"}
                {documentTitle(detail, qrel.document_id) === null && (
                  <span className="table-subcopy">{qrel.document_id}</span>
                )}
              </th>
              <td>
                {relevanceLabel(qrel.relevance_grade)}
                <span className="table-subcopy">Grade {qrel.relevance_grade}</span>
              </td>
              <td>{finalRank(left, qrel.document_id) ?? "Not in top 50"}</td>
              <td>{finalRank(right, qrel.document_id) ?? "Not in top 50"}</td>
              <td>
                <button type="button" onClick={(event) => onInspect(qrel.document_id, event.currentTarget)}>
                  Inspect document
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RankChanges({ detail }: { detail: QueryDetail }) {
  return (
    <div className="rank-change-groups">
      {detail.rank_changes.map((group) => (
        <section key={group.candidate_config_id} aria-labelledby={`rank-change-${group.candidate_config_id}`}>
          <h3 id={`rank-change-${group.candidate_config_id}`}>
            {configName(detail, group.candidate_config_id)} vs baseline
          </h3>
          {group.changes.length === 0 ? (
            <p>No relevant rank changes were recorded for this pair.</p>
          ) : (
            <ul>
              {group.changes.map((change) => (
                <li key={change.document_id}>
                  <strong>{documentTitle(detail, change.document_id) ?? "Title unavailable"}</strong>
                  <span>{relevanceLabel(change.relevance_grade)} · grade {change.relevance_grade}</span>
                  <span>baseline {change.baseline_rank ?? "outside top 50"}</span>
                  <span>candidate {change.candidate_rank ?? "outside top 50"}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      ))}
    </div>
  );
}

function scoreDescription(score: ReplayScore | undefined): string {
  if (score == null) return "Score unavailable";
  return `${score.value.toLocaleString(undefined, { maximumFractionDigits: 6 })} · ${score.kind.replaceAll("_", " ")}`;
}

function DrawerEvidence({
  detail,
  replay,
  selection,
}: {
  detail: QueryDetail;
  replay: ReplayResult | undefined;
  selection: ForensicSelection & { document: string };
}) {
  const qrel = detail.query.qrels.find((item) => item.document_id === selection.document);
  const pair = [selection.left, selection.right];
  const targetObservations = replay?.observations.filter(
    (item) => item.document_id === selection.document && pair.includes(item.config_id),
  ) ?? [];
  const primaryObservations = targetObservations.filter((item) => item.origin === "live_replay_primary");
  const probeObservations = targetObservations.filter(
    (item) => item.origin === "live_replay_counterfactual_probe",
  );
  const computedObservations = targetObservations.filter((item) => item.origin === "client_computed");
  const storedObservations = targetObservations.filter((item) => item.origin === "stored_run");
  const successfulProbes = replay?.counterfactual_probes.filter((probe) => pair.includes(probe.config_id)) ?? [];
  const failedProbes = (replay?.failed_counterfactual_probes ?? []).filter((probe) => pair.includes(probe.config_id));

  return (
    <>
      <section className="drawer-section" aria-labelledby="stored-evidence-heading">
        <h3 id="stored-evidence-heading">Stored run evidence</h3>
        <p>
          Judgment: {qrel === undefined
            ? "not recorded"
            : `${relevanceLabel(qrel.relevance_grade)} (grade ${qrel.relevance_grade})`}.
          {" "}Recorded final ranks:
          {" "}{configName(detail, selection.left)} {finalRank(outcomeFor(detail, selection.left), selection.document) ?? "outside top 50"};
          {" "}{configName(detail, selection.right)} {finalRank(outcomeFor(detail, selection.right), selection.document) ?? "outside top 50"}.
        </p>
        <div className="not-observable" role="note">
          <strong>NOT_OBSERVABLE · original stages</strong>
          <span>
            The stored run did not persist stage membership, stage scores, provider plan, or cache state.
            Final ranks and metrics remain recorded evidence; they are not stage proof.
          </span>
        </div>
        {storedObservations.length > 0 && <ForensicEvidence observations={storedObservations} />}
      </section>

      {replay !== undefined && (
        <>
          <section className="drawer-section" aria-labelledby="primary-evidence-heading">
            <h3 id="primary-evidence-heading">New primary replay</h3>
            {replay.primary.results.filter((result) => pair.includes(result.config.id)).map((result) => {
              const hit = result.hits.find((item) => item.document_id === selection.document);
              return (
                <article className="source-evidence-card" key={result.config.id}>
                  <h4>{configLabel(result.config)}</h4>
                  <p>Trace {result.trace_id} · observed {formatDate(replay.primary_observed_at)}</p>
                  {hit === undefined ? (
                    <p>Document absent from this primary result set.</p>
                  ) : (
                    <>
                      <p>Final rank {hit.final_rank} · {scoreDescription(hit.final_score)}</p>
                      {hit.stage_membership.length > 0 && (
                        <ul>
                          {hit.stage_membership.map((membership) => (
                            <li key={membership.stage}>
                              {membership.stage.replaceAll("_", " ")} rank {membership.rank} · {scoreDescription(membership.score)}
                            </li>
                          ))}
                        </ul>
                      )}
                    </>
                  )}
                </article>
              );
            })}
            {primaryObservations.length > 0 && <ForensicEvidence observations={primaryObservations} />}
          </section>

          <section className="drawer-section counterfactual-section" aria-labelledby="probe-evidence-heading">
            <h3 id="probe-evidence-heading">Counterfactual probe · separate snapshot</h3>
            <p>
              These probes observed a separate provider snapshot. They do not establish the reason for
              the primary ordering, and their timing is never added to primary latency.
            </p>
            {successfulProbes.map((probe) => {
              const candidate = probe.candidates.find((item) => item.document_id === selection.document);
              return (
                <article className="source-evidence-card" key={probe.trace_id}>
                  <h4>{configName(detail, probe.config_id)} · successful probe</h4>
                  <p>
                    {probe.duration_ms.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms separate
                    client wall · {formatDate(probe.observed_at)} · trace {probe.trace_id}
                  </p>
                  <p>{probe.bm25_candidate_count} lexical · {probe.vector_candidate_count} vector candidates</p>
                  {candidate === undefined ? (
                    <p>Document absent from the returned bounded probe candidates.</p>
                  ) : (
                    <ul>
                      {candidate.stage_membership.map((membership) => (
                        <li key={membership.stage}>
                          {membership.stage.replaceAll("_", " ")} rank {membership.rank} · {scoreDescription(membership.score)}
                        </li>
                      ))}
                    </ul>
                  )}
                </article>
              );
            })}
            {failedProbes.map((probe) => (
              <article className="source-evidence-card failed-probe" key={probe.trace_id}>
                <h4>{configName(detail, probe.config_id)} · probe unavailable</h4>
                <p>{probe.warning.message}</p>
                <p>{formatDate(probe.observed_at)} · trace {probe.trace_id}</p>
              </article>
            ))}
            {probeObservations.length > 0 && <ForensicEvidence observations={probeObservations} />}
          </section>

          {computedObservations.length > 0 && (
            <section className="drawer-section" aria-labelledby="computed-evidence-heading">
              <h3 id="computed-evidence-heading">Client-computed arithmetic</h3>
              <ForensicEvidence observations={computedObservations} />
            </section>
          )}
          {targetObservations.length === 0 && <ForensicEvidence observations={[]} />}
        </>
      )}
    </>
  );
}

export function QueryDetailPage({
  runId,
  queryId,
  routeKey,
  routeKind,
  search,
}: {
  runId: string;
  queryId: string;
  routeKey: string;
  routeKind: ForensicRouteKind;
  search: string;
}) {
  const [includeProbe, setIncludeProbe] = useState(false);
  const replayController = useRef<AbortController | null>(null);
  const openerRef = useRef<HTMLElement | null>(null);
  const detailQuery = useQuery({
    queryKey: ["evaluation-run-query", runId, queryId],
    queryFn: ({ signal }) => getEvaluationRunQuery(runId, queryId, signal),
    retry: false,
  });
  const replay = useMutation({
    mutationFn: ({ request, signal }: { request: EvaluationRunQueryReplayRequest; signal: AbortSignal }) =>
      replayEvaluationRunQuery(runId, queryId, request, signal),
  });
  const detail = detailQuery.data;
  const selection = useMemo(
    () => detail === undefined
      ? null
      : resolveForensicSelection(search, detail.baseline_config_id, detail.candidate_config_ids),
    [detail, search],
  );
  const identity = useMemo(() => ({ runId, queryId }), [queryId, runId]);
  const canonicalHref = selection === null ? null : forensicHref(routeKind, identity, selection);
  const selectionKey = selection === null ? null : `${selection.left}:${selection.right}`;
  const previousSelectionKey = useRef<string | null>(null);
  const resetReplay = replay.reset;

  useEffect(() => {
    if (canonicalHref !== null && `${window.location.pathname}${search}` !== canonicalHref) {
      navigate(canonicalHref, { replace: true });
    }
  }, [canonicalHref, search]);

  useEffect(() => {
    if (
      previousSelectionKey.current !== null &&
      selectionKey !== null &&
      previousSelectionKey.current !== selectionKey
    ) {
      replayController.current?.abort();
      resetReplay();
    }
    previousSelectionKey.current = selectionKey;
  }, [resetReplay, selectionKey]);

  useEffect(() => () => replayController.current?.abort(), []);

  function updateSelection(next: ForensicSelection) {
    const pairChanged =
      selection === null || next.left !== selection.left || next.right !== selection.right;
    if (pairChanged) {
      replayController.current?.abort();
      resetReplay();
    }
    navigate(forensicHref(routeKind, identity, next));
  }

  function inspectDocument(documentId: string, trigger: HTMLButtonElement) {
    if (selection === null) return;
    openerRef.current = trigger;
    updateSelection({ ...selection, document: documentId });
  }

  function closeDrawer() {
    if (selection === null) return;
    updateSelection({ ...selection, document: null });
  }

  function submitReplay(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    if (selection === null || detail === undefined || detail.data_origin !== "live" || !detail.live_replay_policy_permitted) {
      return;
    }
    replayController.current?.abort();
    const controller = new AbortController();
    replayController.current = controller;
    replay.mutate({
      request: {
        contract_version: 1,
        config_ids: [selection.left, selection.right],
        include_counterfactual_probe: includeProbe,
      },
      signal: controller.signal,
    });
  }

  const detailError = detailQuery.error instanceof ApiRequestError ? detailQuery.error : null;
  const replayError = replay.error instanceof ApiRequestError ? replay.error : null;

  return (
    <section className="dashboard-page query-detail-page">
      <div className="page-heading compact-heading">
        <p className="eyebrow">Stored query and optional live replay</p>
        <RouteHeading routeKey={routeKey} id="query-forensics-heading">
          {detailError?.status === 404 ? "Query not found" : "Query forensics"}
        </RouteHeading>
        <p className="page-intro">
          Stored evidence loads provider-free. Live replay runs only when explicitly requested.
        </p>
      </div>

      {detailQuery.isPending && <p className="route-loading" role="status">Loading recorded query…</p>}
      {detailQuery.isError && detailError?.status === 404 && (
        <div className="dashboard-empty" role="alert">
          <h2>No query matches this run-scoped URL</h2>
          <p>The query or run may not exist in this local database.</p>
          <AppLink href={`/runs/${encodeURIComponent(runId)}`}>Return to the evaluation run</AppLink>
        </div>
      )}
      {detailQuery.isError && detailError?.status !== 404 && (
        <RequestErrorPanel
          error={detailQuery.error}
          heading="The recorded query is unavailable."
          onRetry={() => void detailQuery.refetch()}
        />
      )}

      {detail !== undefined && selection !== null && (
        <>
          <section className="query-record-panel" aria-labelledby="recorded-query-heading">
            <div className="section-heading-row">
              <div>
                <p className="eyebrow">Stored run · provider-free</p>
                <h2 id="recorded-query-heading">{detail.query.text}</h2>
              </div>
              <span>{detail.data_origin === "synthetic_demo" ? "Synthetic demo · read-only" : "Live recorded run"}</span>
            </div>
            <dl className="query-metadata">
              <div><dt>Query ID</dt><dd>{detail.query.id}</dd></div>
              <div><dt>External ID</dt><dd>{detail.query.external_id}</dd></div>
              <div><dt>Source</dt><dd>{detail.attribution.source_name}</dd></div>
              <div><dt>License</dt><dd>{detail.attribution.license_name ?? "Not specified"}</dd></div>
            </dl>
            <div className="attribution-links">
              {safeSourceUrl(detail.attribution.source_url) !== null && (
                <a href={safeSourceUrl(detail.attribution.source_url) ?? undefined} target="_blank" rel="noreferrer">Dataset source</a>
              )}
              {safeSourceUrl(detail.attribution.license_url) !== null && (
                <a href={safeSourceUrl(detail.attribution.license_url) ?? undefined} target="_blank" rel="noreferrer">Dataset license</a>
              )}
              <AppLink href={`/runs/${encodeURIComponent(runId)}`}>Back to run</AppLink>
            </div>
            <p className="licensed-data-notice" role="note">
              Query text, document titles, and judgments are local licensed data. The canonical URL
              stores UUIDs only.
            </p>
          </section>

          <section className="query-evidence-panel" aria-labelledby="durable-outcomes-heading">
            <div className="section-heading-row">
              <div><p className="eyebrow">Stored evidence</p><h2 id="durable-outcomes-heading">Recorded outcomes</h2></div>
            </div>
            <StoredOutcomes detail={detail} />
            <div className="not-observable" role="note">
              <strong>NOT_OBSERVABLE · original stages</strong>
              <span>
                The stored run preserves final outcomes, ranks, metrics, and judgments, but not stage
                membership, stage scores, provider plan, or cache state. A new replay is a separate
                observation and cannot reconstruct those original stages.
              </span>
            </div>
          </section>

          <section className="query-evidence-panel" aria-labelledby="judgments-heading">
            <div className="section-heading-row">
              <div><p className="eyebrow">Relevance judgments</p><h2 id="judgments-heading">Judged documents</h2></div>
            </div>
            <JudgedDocuments detail={detail} selection={selection} onInspect={inspectDocument} />
            <RankChanges detail={detail} />
          </section>

          <section className="replay-panel" aria-labelledby="live-replay-heading">
            <div className="section-heading-row">
              <div><p className="eyebrow">Explicit new provider request</p><h2 id="live-replay-heading">Live replay</h2></div>
            </div>
            {detail.data_origin === "synthetic_demo" ? (
              <p className="synthetic-notice" role="note">
                <strong>Synthetic demo · replay disabled.</strong> No provider request will be sent.
              </p>
            ) : !detail.live_replay_policy_permitted ? (
              <p className="evidence-policy" role="note">
                Replay is disabled by origin policy. Provider and namespace readiness are not inferred.
              </p>
            ) : (
              <form className="replay-controls" onSubmit={submitReplay}>
                <div className="replay-pair-controls">
                  <label>
                    <span>Left config</span>
                    <select
                      value={selection.left}
                      disabled={replay.isPending}
                      onChange={(event) => {
                        const left = event.target.value;
                        const right = left === selection.right
                          ? detail.configs.find((config) => config.id !== left)?.id ?? selection.right
                          : selection.right;
                        updateSelection({ ...selection, left, right });
                      }}
                    >
                      {detail.configs.map((config) => <option key={config.id} value={config.id}>{configLabel(config)}</option>)}
                    </select>
                  </label>
                  <label>
                    <span>Right config</span>
                    <select
                      value={selection.right}
                      disabled={replay.isPending}
                      onChange={(event) => updateSelection({ ...selection, right: event.target.value })}
                    >
                      {detail.configs.filter((config) => config.id !== selection.left).map((config) => (
                        <option key={config.id} value={config.id}>{configLabel(config)}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <label className="probe-option">
                  <input
                    type="checkbox"
                    checked={includeProbe}
                    disabled={replay.isPending}
                    onChange={(event) => setIncludeProbe(event.target.checked)}
                  />
                  Include separate counterfactual provenance probes (additional provider work)
                </label>
                <p className="cost-notice">
                  Replay is cost-bearing and produces a new request-scoped observation. It does not alter
                  or reconstruct the original run.
                </p>
                <button className="primary-action" type="submit" disabled={replay.isPending}>
                  {replay.isPending ? "Running live replay…" : "Run live replay (cost-bearing)"}
                </button>
                <div className="visually-hidden" role="status" aria-live="polite">
                  {replay.isPending && "Live replay is loading."}
                  {replay.isSuccess && "New live replay observation loaded."}
                </div>
              </form>
            )}

            {replay.isError && (
              <div className="replay-error" role="alert">
                <strong>
                  {replayError?.detail.code === "namespace_not_ready"
                    ? "NOT_OBSERVABLE · provider namespace unavailable"
                    : "The live replay could not be completed."}
                </strong>
                <p>{replayError?.detail.message ?? "An unexpected browser error occurred."}</p>
                {replayError !== null && <p className="trace-id">Trace {replayError.detail.trace_id}</p>}
                <p>The stored query evidence above remains available and unchanged.</p>
                <button type="button" onClick={() => submitReplay()}>Retry explicit replay</button>
              </div>
            )}

            {replay.isSuccess && (
              <section className="replay-results" aria-labelledby="new-observation-heading">
                <div className="replay-origin-notice" role="note">
                  <strong id="new-observation-heading">New live replay · not original run evidence</strong>
                  <span>{formatDate(replay.data.primary_observed_at)} · primary source traces remain per configuration</span>
                  <span>{replay.data.observability_notice}</span>
                </div>
                {(replay.data.failed_counterfactual_probes ?? []).length > 0 && (
                  <ul className="failed-probe-summary" aria-label="Failed counterfactual probes">
                    {(replay.data.failed_counterfactual_probes ?? []).map((probe) => (
                      <li key={probe.trace_id}>
                        <strong>{configName(detail, probe.config_id)} probe unavailable</strong>
                        <span>{probe.warning.message} · {formatDate(probe.observed_at)} · trace {probe.trace_id}</span>
                      </li>
                    ))}
                  </ul>
                )}
                <ComparisonResults
                  response={replay.data.primary}
                  selectedDocumentId={selection.document}
                  onInspectDocument={inspectDocument}
                />
              </section>
            )}
          </section>

          {selection.document !== null && (
            <ForensicDrawer
              documentLabel={documentTitle(detail, selection.document) ?? selection.document}
              onClose={closeDrawer}
              returnFocusRef={openerRef}
            >
              <DrawerEvidence
                detail={detail}
                replay={replay.isSuccess ? replay.data : undefined}
                selection={{ ...selection, document: selection.document }}
              />
              <ExpectedDocumentDiagnostic
                key={JSON.stringify({
                  runId,
                  queryId,
                  documentId: selection.document,
                  relevanceGrade: detail.query.qrels.find(
                    (qrel) => qrel.document_id === selection.document,
                  )?.relevance_grade ?? null,
                  dataOrigin: detail.data_origin,
                  policyPermitted: detail.live_replay_policy_permitted,
                  hasStoredFilter: detail.query.filters !== null,
                  configs: detail.configs,
                })}
                runId={runId}
                queryId={queryId}
                documentId={selection.document}
                relevanceGrade={detail.query.qrels.find(
                  (qrel) => qrel.document_id === selection.document,
                )?.relevance_grade ?? null}
                dataOrigin={detail.data_origin}
                policyPermitted={detail.live_replay_policy_permitted}
                hasStoredFilter={detail.query.filters !== null}
                configs={detail.configs}
              />
            </ForensicDrawer>
          )}
        </>
      )}
    </section>
  );
}
