import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { ApiRequestError } from "../../api/client";
import {
  diagnoseExpectedDocument,
  type EvaluationRunQueryDetailResponse,
  type ExpectedDocumentDiagnosticRequest,
  type ExpectedDocumentDiagnosticResponse,
} from "../../api/evaluations";
import { configLabel } from "../configLabels";
import { ForensicEvidence } from "./ForensicEvidence";
import { diagnosticSubqueryCount } from "./diagnosticPolicy";
import { formatDate } from "./formatters";

type QueryConfig = EvaluationRunQueryDetailResponse["configs"][number];
type DiagnosticScore = NonNullable<ExpectedDocumentDiagnosticResponse["target"]["bm25_score"]>;

type DiagnosticState =
  | { status: "idle" }
  | { status: "pending" }
  | { status: "success"; response: ExpectedDocumentDiagnosticResponse }
  | { status: "error"; error: Error };

function enumLabel(value: string): string {
  return value.replaceAll("_", " ");
}

function scoreText(score: DiagnosticScore | null | undefined): string {
  if (score == null) return "Unavailable";
  return `${score.value.toLocaleString(undefined, { maximumFractionDigits: 8 })} · ${enumLabel(score.kind)}`;
}

function responseMatches(
  response: ExpectedDocumentDiagnosticResponse,
  runId: string,
  queryId: string,
  documentId: string,
  config: QueryConfig,
  includeNoFilterCounterfactual: boolean,
): boolean {
  const sourceMatches = (
    value: {
      config_id: string;
      observed_at: string;
      target_document_id: string;
      trace_id: string;
    },
  ) => value.config_id === response.config_id
    && value.target_document_id === response.target_document_id
    && value.observed_at === response.observed_at
    && value.trace_id === response.trace_id;
  const observationSourcesMatch = response.observations.every((observation) =>
    observation.config_id === response.config_id
    && observation.document_id === response.target_document_id
    && observation.observed_at === response.observed_at
    && observation.trace_id === response.trace_id
    && (observation.origin === "live_expected_document_diagnostic" || observation.origin === "client_computed")
    && observation.evidence.every((item) => {
      const originMatches = observation.origin === "live_expected_document_diagnostic"
        ? item.origin === "live_expected_document_diagnostic"
        : item.origin === "live_expected_document_diagnostic" || item.origin === "client_computed";
      return item.observed_at === response.observed_at
        && item.trace_id === response.trace_id
        && originMatches;
    }),
  );
  return response.contract_version === 1
    && response.run_id === runId
    && response.query_id === queryId
    && response.target_document_id === documentId
    && response.config_id === config.id
    && response.config_mode === config.mode
    && response.included_no_filter_counterfactual === includeNoFilterCounterfactual
    && response.data_origin === "live"
    && response.origin === "live_expected_document_diagnostic"
    && response.observability_notice === "new_live_diagnostic_not_original_run"
    && sourceMatches(response.target)
    && response.target.origin === "live_expected_document_diagnostic"
    && response.filter_evidence.every((item) => sourceMatches(item) && item.origin === "client_computed")
    && response.candidate_evidence.every((item) => sourceMatches(item) && item.origin === "client_computed")
    && response.qualified_rrf_evidence.every((item) => sourceMatches(item) && item.origin === "client_computed")
    && observationSourcesMatch;
}

function SubqueryEvidence({ response }: { response: ExpectedDocumentDiagnosticResponse }) {
  return (
    <section className="diagnostic-evidence-group" aria-labelledby="diagnostic-subqueries-heading">
      <h4 id="diagnostic-subqueries-heading">Bounded subqueries</h4>
      <ol className="diagnostic-card-list">
        {response.subqueries.map((subquery) => (
          <li key={`${subquery.ordinal}-${subquery.role}`}>
            <strong>{subquery.ordinal + 1}. {enumLabel(subquery.role)}</strong>
            <dl className="diagnostic-facts">
              <div><dt>Requested limit</dt><dd>{subquery.requested_limit}</dd></div>
              <div><dt>Returned count</dt><dd>{subquery.returned_count}</dd></div>
              <div><dt>Selected target</dt><dd>{subquery.target_present ? "Present" : "Absent"}</dd></div>
              {subquery.kind === "candidate" && (
                <>
                  <div><dt>Target rank</dt><dd>{subquery.target_rank ?? "Unavailable"}</dd></div>
                  <div><dt>Target score</dt><dd>{scoreText(subquery.target_score)}</dd></div>
                  <div><dt>Full-list boundary</dt><dd>{scoreText(subquery.boundary_score)}</dd></div>
                </>
              )}
            </dl>
          </li>
        ))}
      </ol>
    </section>
  );
}

function TargetEvidence({ response }: { response: ExpectedDocumentDiagnosticResponse }) {
  return (
    <section className="diagnostic-evidence-group" aria-labelledby="diagnostic-target-heading">
      <h4 id="diagnostic-target-heading">Exact target lookup</h4>
      {response.target.available ? (
        <dl className="diagnostic-facts">
          <div><dt>Availability</dt><dd>Available in this diagnostic snapshot</dd></div>
          {response.target.bm25_score != null && (
            <div><dt>Direct BM25 score</dt><dd>{scoreText(response.target.bm25_score)}</dd></div>
          )}
          {response.target.vector_distance != null && (
            <div><dt>Direct VectorDist</dt><dd>{scoreText(response.target.vector_distance)}</dd></div>
          )}
        </dl>
      ) : (
        <div className="not-observable" role="note">
          <strong>NOT_OBSERVABLE · target unavailable</strong>
          <span>
            The selected target was unavailable in this diagnostic snapshot. This does not change its
            authenticated judgment or the stored run.
          </span>
        </div>
      )}
    </section>
  );
}

function FilterEvidence({ response }: { response: ExpectedDocumentDiagnosticResponse }) {
  if (response.filter_evidence.length === 0) {
    return (
      <section className="diagnostic-evidence-group" aria-labelledby="diagnostic-filter-heading">
        <h4 id="diagnostic-filter-heading">Stored-query filter</h4>
        <p>No stored query filter was evaluated for this diagnostic.</p>
      </section>
    );
  }
  return (
    <section className="diagnostic-evidence-group" aria-labelledby="diagnostic-filter-heading">
      <h4 id="diagnostic-filter-heading">Stored filter · evaluated locally</h4>
      <p className="diagnostic-aggregate">
        Aggregate result: <strong>{enumLabel(response.stored_filter_result ?? "not_observable")}</strong>
      </p>
      <ol className="diagnostic-card-list">
        {response.filter_evidence.map((item) => (
          <li key={`${item.predicate_ordinal}-${item.predicate_path.join(".")}`}>
            <strong>Predicate {item.predicate_ordinal + 1} · {enumLabel(item.result)}</strong>
            <p>
              Path {item.predicate_path.join(".")} · field {item.field} · {enumLabel(item.operator)} ·
              {" "}{enumLabel(item.certainty)}
            </p>
          </li>
        ))}
      </ol>
      <p className="diagnostic-limit-copy">
        Predicate and observed attribute values are intentionally omitted.
      </p>
    </section>
  );
}

function CandidateScope({
  response,
  scope,
}: {
  response: ExpectedDocumentDiagnosticResponse;
  scope: ExpectedDocumentDiagnosticResponse["candidate_evidence"][number]["scope"];
}) {
  const evidence = response.candidate_evidence.filter((item) => item.scope === scope);
  const counterfactual = scope === "no_filter_counterfactual";
  if (evidence.length === 0) return null;
  return (
    <section
      className={`diagnostic-evidence-group ${counterfactual ? "counterfactual-section" : ""}`}
      aria-labelledby={`diagnostic-candidates-${scope}`}
    >
      <h4 id={`diagnostic-candidates-${scope}`}>
        {counterfactual ? "No-filter counterfactual candidates" : "Stored-query candidates"}
      </h4>
      {counterfactual && (
        <p>
          These facts are counterfactual inputs from the same diagnostic request. They do not establish
          why the stored-query candidate lists differ.
        </p>
      )}
      <ul className="diagnostic-card-list">
        {evidence.map((item) => (
          <li key={`${item.subquery_ordinal}-${item.role}`}>
            <strong>{item.signal.toUpperCase()} · {enumLabel(item.relation)}</strong>
            <dl className="diagnostic-facts">
              <div><dt>Returned</dt><dd>{item.returned_count} of limit {item.requested_limit}</dd></div>
              <div><dt>Selected target</dt><dd>{item.target_present ? "Present" : "Absent"}</dd></div>
              <div><dt>Target rank</dt><dd>{item.target_rank ?? "Unavailable"}</dd></div>
              <div><dt>Direct score</dt><dd>{scoreText(item.direct_score)}</dd></div>
              <div><dt>Candidate score</dt><dd>{scoreText(item.target_score)}</dd></div>
              <div><dt>Full-list boundary</dt><dd>{scoreText(item.boundary_score)}</dd></div>
              <div><dt>Certainty</dt><dd>{enumLabel(item.certainty)}</dd></div>
            </dl>
          </li>
        ))}
      </ul>
    </section>
  );
}

function QualifiedRrfEvidence({ response }: { response: ExpectedDocumentDiagnosticResponse }) {
  if (response.qualified_rrf_evidence.length === 0) return null;
  return (
    <section className="diagnostic-evidence-group" aria-labelledby="diagnostic-rrf-heading">
      <h4 id="diagnostic-rrf-heading">Client-computed RRF</h4>
      <p>
        This arithmetic uses only returned bounded inputs. It is not observed server RRF, reranker,
        or final-order evidence.
      </p>
      <ul className="diagnostic-card-list">
        {response.qualified_rrf_evidence.map((item) => (
          <li className={item.scope === "no_filter_counterfactual" ? "is-counterfactual" : undefined} key={item.scope}>
            <strong>{enumLabel(item.scope)} · {enumLabel(item.relation)}</strong>
            <dl className="diagnostic-facts">
              <div><dt>BM25 input</dt><dd>rank {item.bm25_rank ?? "absent"} · weight {item.bm25_weight}</dd></div>
              <div><dt>ANN input</dt><dd>rank {item.ann_rank ?? "absent"} · weight {item.ann_weight}</dd></div>
              <div><dt>Rank constant</dt><dd>{item.rank_constant}</dd></div>
              <div><dt>Qualified list</dt><dd>{item.returned_count} of cutoff {item.cutoff}</dd></div>
              <div><dt>Selected target</dt><dd>{item.target_present ? `Present at rank ${item.target_rank}` : "Absent"}</dd></div>
              <div><dt>Target RRF score</dt><dd>{scoreText(item.target_score)}</dd></div>
              <div><dt>Boundary RRF score</dt><dd>{scoreText(item.boundary_score)}</dd></div>
              <div><dt>Certainty</dt><dd>{enumLabel(item.certainty)}</dd></div>
            </dl>
          </li>
        ))}
      </ul>
    </section>
  );
}

function DiagnosticResult({
  response,
  config,
}: {
  response: ExpectedDocumentDiagnosticResponse;
  config: QueryConfig;
}) {
  return (
    <section className="diagnostic-results" aria-labelledby="diagnostic-result-heading">
      <div className="diagnostic-origin-notice" role="note">
        <strong id="diagnostic-result-heading">New live expected-document diagnostic · not original run evidence</strong>
        <span>{configLabel(config)} · {response.subqueries.length} ordered subqueries</span>
        <span>{formatDate(response.observed_at)} · trace {response.trace_id}</span>
        <span>
          {response.duration_ms.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms total
          {response.embedding_duration_ms == null
            ? " · no query embedding"
            : ` · ${response.embedding_duration_ms.toLocaleString(undefined, { maximumFractionDigits: 1 })} ms embedding`}
        </span>
      </div>
      <TargetEvidence response={response} />
      <SubqueryEvidence response={response} />
      <FilterEvidence response={response} />
      <CandidateScope response={response} scope="stored_query" />
      <CandidateScope response={response} scope="no_filter_counterfactual" />
      <QualifiedRrfEvidence response={response} />
      <section className="diagnostic-evidence-group" aria-labelledby="diagnostic-findings-heading">
        <h4 id="diagnostic-findings-heading">Findings and limits</h4>
        {response.observations.length > 0 ? (
          <ForensicEvidence observations={response.observations} />
        ) : (
          <div className="forensic-empty" role="note">
            <strong>NO ADDITIONAL FINDING</strong>
            <span>The target-scoped facts above did not require another typed finding.</span>
          </div>
        )}
      </section>
    </section>
  );
}

type ExpectedDocumentDiagnosticProps = {
  runId: string;
  queryId: string;
  documentId: string;
  relevanceGrade: number | null;
  dataOrigin: EvaluationRunQueryDetailResponse["data_origin"];
  policyPermitted: boolean;
  hasStoredFilter: boolean;
  configs: EvaluationRunQueryDetailResponse["configs"];
};

function DiagnosticSession({
  runId,
  queryId,
  documentId,
  relevanceGrade,
  dataOrigin,
  policyPermitted,
  hasStoredFilter,
  configs,
}: ExpectedDocumentDiagnosticProps) {
  const [configId, setConfigId] = useState("");
  const [includeNoFilter, setIncludeNoFilter] = useState(false);
  const [costConfirmed, setCostConfirmed] = useState(false);
  const [state, setState] = useState<DiagnosticState>({ status: "idle" });
  const controllerRef = useRef<AbortController | null>(null);
  const epochRef = useRef(0);
  const selectedConfig = useMemo(
    () => configs.find((config) => config.id === configId),
    [configId, configs],
  );
  const eligible = dataOrigin === "live" && policyPermitted && relevanceGrade !== null && relevanceGrade > 0;

  function invalidateRequest() {
    epochRef.current += 1;
    controllerRef.current?.abort();
    controllerRef.current = null;
    setState({ status: "idle" });
    setCostConfirmed(false);
  }

  useEffect(() => () => {
    epochRef.current += 1;
    controllerRef.current?.abort();
  }, []);

  function handleConfigChange(nextConfigId: string) {
    invalidateRequest();
    setConfigId(nextConfigId);
  }

  function handleNoFilterChange(nextValue: boolean) {
    invalidateRequest();
    setIncludeNoFilter(nextValue);
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!eligible || selectedConfig === undefined || !costConfirmed) return;
    invalidateRequest();
    const requestEpoch = epochRef.current;
    const requestConfig = selectedConfig;
    const requestIncludeNoFilter = hasStoredFilter && includeNoFilter;
    const controller = new AbortController();
    controllerRef.current = controller;
    setState({ status: "pending" });
    const request: ExpectedDocumentDiagnosticRequest = {
      contract_version: 1,
      config_id: requestConfig.id,
      include_no_filter_counterfactual: requestIncludeNoFilter,
    };
    void diagnoseExpectedDocument(runId, queryId, documentId, request, controller.signal)
      .then((response) => {
        if (epochRef.current !== requestEpoch || controller.signal.aborted) return;
        if (!responseMatches(
          response,
          runId,
          queryId,
          documentId,
          requestConfig,
          requestIncludeNoFilter,
        )) {
          setState({ status: "error", error: new Error("diagnostic_response_mismatch") });
          return;
        }
        setState({ status: "success", response });
      })
      .catch((error: unknown) => {
        if (epochRef.current !== requestEpoch || controller.signal.aborted) return;
        setState({
          status: "error",
          error: error instanceof Error ? error : new Error("diagnostic_request_failed"),
        });
      })
      .finally(() => {
        if (epochRef.current === requestEpoch && controllerRef.current === controller) {
          controllerRef.current = null;
        }
      });
  }

  const subqueryCount = selectedConfig === undefined
    ? null
    : diagnosticSubqueryCount(selectedConfig.mode, includeNoFilter);
  const requestError = state.status === "error" && state.error instanceof ApiRequestError
    ? state.error
    : null;

  return (
    <section className="drawer-section diagnostic-section" aria-labelledby="expected-document-diagnostic-heading">
      <p className="eyebrow">Explicit cost-bearing observation</p>
      <h3 id="expected-document-diagnostic-heading">Expected-document diagnostic</h3>
      <p>
        Inspect this positively judged document in one new provider snapshot. Opening the drawer or
        changing controls does nothing.
      </p>

      {!eligible ? (
        <div className="evidence-policy" role="note">
          <strong>Diagnostic unavailable.</strong>{" "}
          {dataOrigin !== "live" || !policyPermitted
            ? "Only authenticated live recorded runs permit this cost-bearing action."
            : relevanceGrade === null
              ? "The selected document is not an authenticated qrel for this query."
              : "Only a positively judged document is eligible."}
        </div>
      ) : (
        <form className="diagnostic-controls" onSubmit={submit}>
          <label>
            <span>Configuration</span>
            <select
              value={configId}
              onChange={(event) => handleConfigChange(event.target.value)}
            >
              <option value="">Choose a configuration</option>
              {configs.map((config) => (
                <option key={config.id} value={config.id}>{configLabel(config)}</option>
              ))}
            </select>
          </label>

          <label className="diagnostic-option">
            <input
              type="checkbox"
              checked={includeNoFilter}
              disabled={!hasStoredFilter}
              onChange={(event) => handleNoFilterChange(event.target.checked)}
            />
            Include a same-request no-filter counterfactual
          </label>
          {!hasStoredFilter && (
            <p className="diagnostic-limit-copy">No stored query filter exists, so the no-filter option is ineligible.</p>
          )}

          {selectedConfig !== undefined && subqueryCount !== null && (
            <div className="diagnostic-cost-notice" role="note">
              <strong>Exact request bound: one SDK call, at most one HTTP attempt, exactly {subqueryCount} ordered subqueries.</strong>
              <span>
                Cost depends on workload-dependent logical bytes queried and returned and namespace
                configuration; every subquery counts toward the namespace&apos;s concurrent query limit.
                This is a new observation, not the original run.
              </span>
            </div>
          )}

          <label className="diagnostic-option cost-confirmation">
            <input
              type="checkbox"
              checked={costConfirmed}
              disabled={selectedConfig === undefined || state.status === "pending"}
              onChange={(event) => setCostConfirmed(event.target.checked)}
            />
            I understand this starts cost-bearing provider work.
          </label>

          <button
            className="primary-action"
            type="submit"
            disabled={selectedConfig === undefined || !costConfirmed || state.status === "pending"}
          >
            {state.status === "pending" ? "Running expected-document diagnostic…" : "Run expected-document diagnostic"}
          </button>
          <div className="visually-hidden" role="status" aria-live="polite">
            {state.status === "pending" && "Expected-document diagnostic is loading."}
            {state.status === "success" && "New expected-document diagnostic loaded."}
          </div>
        </form>
      )}

      {state.status === "error" && (
        <div className="diagnostic-error" role="alert">
          <strong>The expected-document diagnostic could not be completed.</strong>
          <p>{requestError?.detail.message ?? "The diagnostic response was unavailable or did not match this exact selection."}</p>
          {requestError !== null && <p className="trace-id">Trace {requestError.detail.trace_id}</p>}
          <p>Stored run and separately requested replay evidence remain available and unchanged.</p>
          <p>Review the cost notice, confirm again, and use the explicit run action to retry.</p>
        </div>
      )}

      {state.status === "success" && selectedConfig !== undefined && (
        <DiagnosticResult response={state.response} config={selectedConfig} />
      )}
    </section>
  );
}

export function ExpectedDocumentDiagnostic(props: ExpectedDocumentDiagnosticProps) {
  const sessionKey = JSON.stringify({
    runId: props.runId,
    queryId: props.queryId,
    documentId: props.documentId,
    relevanceGrade: props.relevanceGrade,
    dataOrigin: props.dataOrigin,
    policyPermitted: props.policyPermitted,
    hasStoredFilter: props.hasStoredFilter,
    configs: props.configs,
  });
  return <DiagnosticSession key={sessionKey} {...props} />;
}
