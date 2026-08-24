import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../api/evaluations", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/evaluations")>();
  return { ...actual, diagnoseExpectedDocument: vi.fn() };
});

import { ApiRequestError } from "../../api/client";
import {
  diagnoseExpectedDocument,
  type ExpectedDocumentDiagnosticResponse,
} from "../../api/evaluations";
import {
  baselineId,
  candidateIds,
  diagnosticTrace,
  documentId,
  evaluationConfigs,
  expectedDocumentDiagnosticResponse,
  queryId,
  runId,
} from "../../test/evalFixtures";
import { ExpectedDocumentDiagnostic } from "./ExpectedDocumentDiagnostic";

type Props = Parameters<typeof ExpectedDocumentDiagnostic>[0];

const baseProps: Props = {
  runId,
  queryId,
  documentId,
  relevanceGrade: 2,
  dataOrigin: "live",
  policyPermitted: true,
  hasStoredFilter: true,
  configs: evaluationConfigs,
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function renderDiagnostic(props: Partial<Props> = {}) {
  return render(<ExpectedDocumentDiagnostic {...baseProps} {...props} />);
}

function selectAndConfirm(configId: string, includeNoFilter = false) {
  fireEvent.change(screen.getByLabelText("Diagnostic configuration"), {
    target: { value: configId },
  });
  if (includeNoFilter) {
    fireEvent.click(screen.getByLabelText("Include a same-request no-filter counterfactual"));
  }
  fireEvent.click(screen.getByLabelText("I understand this starts cost-bearing provider work."));
}

function runDiagnostic(configId: string, includeNoFilter = false) {
  selectAndConfirm(configId, includeNoFilter);
  fireEvent.click(screen.getByRole("button", { name: "Run expected-document diagnostic" }));
}

beforeEach(() => {
  vi.mocked(diagnoseExpectedDocument).mockImplementation((_run, _query, _document, request) =>
    Promise.resolve(expectedDocumentDiagnosticResponse(
      request.config_id,
      request.include_no_filter_counterfactual,
    )),
  );
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("ExpectedDocumentDiagnostic", () => {
  it.each([
    [baselineId, 2, 3],
    [candidateIds[0], 2, 3],
    [candidateIds[1], 3, 5],
    [candidateIds[2], 3, 5],
  ] as const)("discloses exact normal and no-filter bounds for config %s", (configId, normal, noFilter) => {
    renderDiagnostic();

    expect(screen.getByLabelText("Diagnostic configuration")).toHaveValue("");
    expect(screen.getByRole("button", { name: "Run expected-document diagnostic" })).toBeDisabled();
    expect(diagnoseExpectedDocument).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText("Diagnostic configuration"), {
      target: { value: configId },
    });
    expect(screen.getByText(new RegExp(`exactly ${normal} ordered subqueries`, "i"))).toBeVisible();
    expect(screen.getByText(/workload-dependent logical bytes queried and returned/i)).toBeVisible();
    expect(screen.getByText(/every subquery counts toward the namespace's concurrent query limit/i)).toBeVisible();

    fireEvent.click(screen.getByLabelText("Include a same-request no-filter counterfactual"));
    expect(screen.getByText(new RegExp(`exactly ${noFilter} ordered subqueries`, "i"))).toBeVisible();
    expect(screen.getByLabelText("I understand this starts cost-bearing provider work.")).not.toBeChecked();
    expect(diagnoseExpectedDocument).not.toHaveBeenCalled();
  });

  it.each([
    [{ dataOrigin: "synthetic_demo" as const }, "Only authenticated live recorded runs"],
    [{ policyPermitted: false }, "Only authenticated live recorded runs"],
    [{ relevanceGrade: 0 }, "Only a positively judged document"],
    [{ relevanceGrade: null }, "not an authenticated qrel"],
  ])("keeps ineligible evidence provider-free", (props, copy) => {
    renderDiagnostic(props);

    expect(screen.getByText(copy, { exact: false })).toBeVisible();
    expect(screen.queryByLabelText("Diagnostic configuration")).not.toBeInTheDocument();
    expect(diagnoseExpectedDocument).not.toHaveBeenCalled();
  });

  it("disables no-filter work when the recorded query has no filter", () => {
    renderDiagnostic({ hasStoredFilter: false });
    const option = screen.getByLabelText("Include a same-request no-filter counterfactual");
    expect(option).toBeDisabled();
    expect(screen.getByText(/no-filter option is ineligible/i)).toBeVisible();
    fireEvent.change(screen.getByLabelText("Diagnostic configuration"), {
      target: { value: baselineId },
    });
    expect(screen.getByText(/exactly 2 ordered subqueries/i)).toBeVisible();
  });

  it.each([
    [baselineId, "BM25", 2],
    [candidateIds[0], "Vector", 2],
    [candidateIds[1], "Hybrid RRF", 3],
    [candidateIds[2], "Hybrid rerank", 3],
  ] as const)("runs %s only after fresh confirmation and renders typed evidence", async (configId, mode, count) => {
    renderDiagnostic();
    runDiagnostic(configId);

    expect(diagnoseExpectedDocument).toHaveBeenCalledWith(
      runId,
      queryId,
      documentId,
      { contract_version: 1, config_id: configId, include_no_filter_counterfactual: false },
      expect.any(AbortSignal),
    );
    const resultHeading = await screen.findByText("New live expected-document diagnostic · not original run evidence");
    const originNotice = resultHeading.closest("[role='note']");
    expect(originNotice).not.toBeNull();
    expect(originNotice).toHaveTextContent(mode);
    expect(originNotice).toHaveTextContent(`${count} ordered subqueries`);
    expect(screen.getByRole("heading", { name: "Exact selected-target lookup" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Stored-query candidates" })).toBeVisible();
    expect(screen.getByLabelText("I understand this starts cost-bearing provider work.")).not.toBeChecked();
    expect(screen.getByRole("button", { name: "Run expected-document diagnostic" })).toBeDisabled();
    if (mode.startsWith("Hybrid")) {
      expect(screen.getByRole("heading", { name: "Qualified client-computed RRF" })).toBeVisible();
      expect(screen.getByText(/not observed server RRF, reranker, or final-order evidence/i)).toBeVisible();
    }
  });

  it("separates same-request no-filter evidence and locally evaluated filter facts", async () => {
    vi.mocked(diagnoseExpectedDocument).mockResolvedValueOnce(
      expectedDocumentDiagnosticResponse(baselineId, true, { storedFilterResult: "not_matched" }),
    );
    renderDiagnostic();
    runDiagnostic(baselineId, true);

    expect(await screen.findByRole("heading", { name: "Same-request no-filter counterfactual candidates" })).toBeVisible();
    expect(screen.getByText(/do not establish why the stored-query candidate lists differ/i)).toBeVisible();
    expect(screen.getByText(/Aggregate result:/)).toHaveTextContent("not matched");
    expect(screen.getByText(/Predicate 1 · not matched/)).toBeVisible();
    expect(screen.getByText(/Predicate and observed attribute values are intentionally omitted/)).toBeVisible();
    expect(screen.getAllByText(/no filter counterfactual/i).length).toBeGreaterThan(0);
    const counterfactual = screen.getByRole("heading", {
      name: "Same-request no-filter counterfactual candidates",
    }).closest("section");
    expect(counterfactual).not.toBeNull();
    expect(within(counterfactual!).getByText("counterfactual")).toBeVisible();
  });

  it("renders unavailable evidence without inventing an explanation", async () => {
    const response = expectedDocumentDiagnosticResponse(candidateIds[0], false, {
      targetAvailable: false,
    });
    vi.mocked(diagnoseExpectedDocument).mockResolvedValueOnce(response);
    renderDiagnostic();
    runDiagnostic(candidateIds[0]);

    expect(await screen.findByText(/NOT_OBSERVABLE · target unavailable/)).toBeVisible();
    expect(screen.getAllByText(/unavailable in this diagnostic snapshot/i).length).toBeGreaterThan(0);
    expect(document.body).not.toHaveTextContent("filter ran before ann");
    expect(document.body).not.toHaveTextContent("cache was cold");
  });

  it.each([
    ["outside cutoff", 0.5, "outside_candidates", "observed"],
    ["tolerance tie", 1 + 5e-16, "not_observable", "insufficient"],
  ] as const)("renders a bounded BM25 %s state from typed facts", async (_label, directValue, relation, certainty) => {
    const response = expectedDocumentDiagnosticResponse();
    const boundaryScore = {
      value: 1,
      kind: "bm25" as const,
      direction: "higher_is_better" as const,
      source: "turbopuffer_dist" as const,
    };
    response.target.bm25_score = {
      ...boundaryScore,
      value: directValue,
      source: "compute_attribute",
    };
    const summary = response.subqueries[1];
    const evidence = response.candidate_evidence[0];
    if (summary?.kind !== "candidate" || evidence === undefined) {
      throw new Error("Expected an authored BM25 candidate fixture");
    }
    Object.assign(summary, {
      returned_count: 50,
      target_present: false,
      target_rank: null,
      target_score: null,
      boundary_score: boundaryScore,
    });
    Object.assign(evidence, {
      returned_count: 50,
      target_present: false,
      target_rank: null,
      target_score: null,
      direct_score: response.target.bm25_score,
      boundary_score: boundaryScore,
      relation,
      certainty,
    });
    response.observations = [{
      config_id: baselineId,
      document_id: documentId,
      code: relation === "outside_candidates" ? "outside_lexical_candidates" : "not_observable",
      statement: relation === "outside_candidates"
        ? "The selected target scored outside the lexical candidate boundary."
        : "The selected target's exclusion is not observable from this diagnostic.",
      origin: "client_computed",
      observed_at: response.observed_at,
      trace_id: response.trace_id,
      certainty,
      evidence: [{
        label: "cutoff_stored_query_bm25",
        value: {
          kind: "diagnostic_cutoff_relation",
          scope: "stored_query",
          signal: "bm25",
          relation,
        },
        origin: "client_computed",
        observed_at: response.observed_at,
        trace_id: response.trace_id,
      }],
    }];
    vi.mocked(diagnoseExpectedDocument).mockResolvedValueOnce(response);
    renderDiagnostic();
    runDiagnostic(baselineId);

    expect((await screen.findAllByText(
      new RegExp(`BM25 · ${relation.replaceAll("_", " ")}`, "i"),
    )).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/1 · bm25 · higher is better · turbopuffer dist/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(new RegExp(certainty, "i")).length).toBeGreaterThan(0);
    expect(document.body).not.toHaveTextContent("filter ran before ann");
  });

  it("aborts and suppresses a late identity success when positive eligibility disappears", async () => {
    const pending = deferred<ExpectedDocumentDiagnosticResponse>();
    let signal: AbortSignal | undefined;
    vi.mocked(diagnoseExpectedDocument).mockImplementation((_run, _query, _document, _request, requestSignal) => {
      signal = requestSignal;
      return pending.promise;
    });
    const view = renderDiagnostic();
    runDiagnostic(baselineId);
    await waitFor(() => expect(signal).toBeDefined());

    view.rerender(<ExpectedDocumentDiagnostic {...baseProps} relevanceGrade={0} />);
    expect(signal?.aborted).toBe(true);
    expect(screen.getByText(/Only a positively judged document/)).toBeVisible();
    pending.resolve(expectedDocumentDiagnosticResponse());
    await Promise.resolve();
    expect(screen.queryByText("New live expected-document diagnostic · not original run evidence")).not.toBeInTheDocument();
  });

  it("aborts and suppresses a late option error while keeping controls enabled for cancellation", async () => {
    const pending = deferred<ExpectedDocumentDiagnosticResponse>();
    let signal: AbortSignal | undefined;
    vi.mocked(diagnoseExpectedDocument).mockImplementation((_run, _query, _document, _request, requestSignal) => {
      signal = requestSignal;
      return pending.promise;
    });
    renderDiagnostic();
    runDiagnostic(candidateIds[1]);
    await waitFor(() => expect(signal).toBeDefined());
    expect(screen.getByLabelText("Diagnostic configuration")).toBeEnabled();
    expect(screen.getByLabelText("Include a same-request no-filter counterfactual")).toBeEnabled();

    fireEvent.click(screen.getByLabelText("Include a same-request no-filter counterfactual"));
    expect(signal?.aborted).toBe(true);
    pending.reject(new Error("late diagnostic failure"));
    await Promise.resolve();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByLabelText("I understand this starts cost-bearing provider work.")).not.toBeChecked();
  });

  it.each([
    ["run", "resolve"],
    ["run", "reject"],
    ["query", "resolve"],
    ["query", "reject"],
    ["document", "resolve"],
    ["document", "reject"],
    ["config-set", "resolve"],
    ["config-set", "reject"],
    ["selected-config", "resolve"],
    ["selected-config", "reject"],
    ["option", "resolve"],
    ["option", "reject"],
  ] as const)("suppresses a late %s-boundary %s without retry", async (boundary, settlement) => {
    const pending = deferred<ExpectedDocumentDiagnosticResponse>();
    let signal: AbortSignal | undefined;
    vi.mocked(diagnoseExpectedDocument).mockImplementation((_run, _query, _document, _request, requestSignal) => {
      signal = requestSignal;
      return pending.promise;
    });
    const view = renderDiagnostic();
    runDiagnostic(baselineId);
    await waitFor(() => expect(signal).toBeDefined());

    switch (boundary) {
      case "run":
        view.rerender(<ExpectedDocumentDiagnostic {...baseProps} runId="31000000-0000-4000-8000-000000000003" />);
        break;
      case "query":
        view.rerender(<ExpectedDocumentDiagnostic {...baseProps} queryId="81000000-0000-4000-8000-000000000008" />);
        break;
      case "document":
        view.rerender(<ExpectedDocumentDiagnostic {...baseProps} documentId="92000000-0000-4000-8000-000000000009" />);
        break;
      case "config-set":
        view.rerender(<ExpectedDocumentDiagnostic {...baseProps} configs={evaluationConfigs.slice(0, 3)} />);
        break;
      case "selected-config":
        fireEvent.change(screen.getByLabelText("Diagnostic configuration"), {
          target: { value: candidateIds[0] },
        });
        break;
      case "option":
        fireEvent.click(screen.getByLabelText("Include a same-request no-filter counterfactual"));
        break;
    }

    expect(signal?.aborted).toBe(true);
    if (settlement === "resolve") pending.resolve(expectedDocumentDiagnosticResponse());
    else pending.reject(new Error("late boundary failure"));
    await waitFor(() => {
      expect(screen.queryByText("New live expected-document diagnostic · not original run evidence")).not.toBeInTheDocument();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });
    expect(diagnoseExpectedDocument).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText("Diagnostic configuration")).toHaveValue(
      boundary === "selected-config" ? candidateIds[0] : boundary === "option" ? baselineId : "",
    );
    expect(screen.getByLabelText("I understand this starts cost-bearing provider work.")).not.toBeChecked();
  });

  it("clears a prior result on rerun and requires confirmation again after failure", async () => {
    const retry = deferred<ExpectedDocumentDiagnosticResponse>();
    vi.mocked(diagnoseExpectedDocument)
      .mockResolvedValueOnce(expectedDocumentDiagnosticResponse())
      .mockImplementationOnce(() => retry.promise)
      .mockRejectedValueOnce(new ApiRequestError({
        code: "namespace_not_ready",
        message: "The diagnostic namespace is unavailable.",
        retryable: true,
        trace_id: "safe-diagnostic-error",
      }, 503));
    renderDiagnostic();
    runDiagnostic(baselineId);
    expect(await screen.findByText("New live expected-document diagnostic · not original run evidence")).toBeVisible();

    selectAndConfirm(baselineId);
    fireEvent.click(screen.getByRole("button", { name: "Run expected-document diagnostic" }));
    expect(screen.queryByText("New live expected-document diagnostic · not original run evidence")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Running expected-document diagnostic/ })).toBeDisabled();
    retry.resolve(expectedDocumentDiagnosticResponse());
    expect(await screen.findByText("New live expected-document diagnostic · not original run evidence")).toBeVisible();

    expect(screen.getByRole("button", { name: "Run expected-document diagnostic" })).toBeDisabled();
    fireEvent.click(screen.getByLabelText("I understand this starts cost-bearing provider work."));
    fireEvent.click(screen.getByRole("button", { name: "Run expected-document diagnostic" }));
    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText("The diagnostic namespace is unavailable.")).toBeVisible();
    expect(within(alert).getByText(/Stored run and separately requested replay evidence remain available/)).toBeVisible();
    expect(screen.getByLabelText("I understand this starts cost-bearing provider work.")).not.toBeChecked();
  });

  it("rejects response echo mismatches and never renders hostile arbitrary fields", async () => {
    const marker = "M5E_HOSTILE_ARBITRARY_MARKER";
    const mismatch = expectedDocumentDiagnosticResponse();
    const hostile = mismatch as ExpectedDocumentDiagnosticResponse & Record<string, unknown>;
    hostile.query_id = "81000000-0000-4000-8000-000000000008";
    hostile.api_key = marker;
    Object.assign(hostile.target, { provider_body: marker });
    Object.assign(hostile.candidate_evidence[0] ?? {}, { namespace: marker });
    vi.mocked(diagnoseExpectedDocument).mockResolvedValueOnce(hostile);
    renderDiagnostic();
    runDiagnostic(baselineId);

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText(/did not match this exact selection/i)).toBeVisible();
    expect(document.body).not.toHaveTextContent(marker);
    expect(document.body).not.toHaveTextContent("81000000-0000-4000-8000-000000000008");
    expect(document.body).not.toHaveTextContent(diagnosticTrace);
  });

  it.each([
    ["target", baselineId],
    ["filter", baselineId],
    ["candidate", baselineId],
    ["rrf", candidateIds[1]],
    ["observation", baselineId],
  ] as const)("rejects a foreign nested %s source before rendering", async (source, configId) => {
    const response = source === "observation"
      ? expectedDocumentDiagnosticResponse(configId, false, { targetAvailable: false })
      : expectedDocumentDiagnosticResponse(
        configId,
        false,
        source === "filter" ? { storedFilterResult: "matched" } : {},
      );
    const foreignTrace = "af000000-0000-4000-8000-00000000000f";
    switch (source) {
      case "target":
        response.target.trace_id = foreignTrace;
        break;
      case "filter":
        if (response.filter_evidence[0] !== undefined) response.filter_evidence[0].trace_id = foreignTrace;
        break;
      case "candidate":
        if (response.candidate_evidence[0] !== undefined) response.candidate_evidence[0].trace_id = foreignTrace;
        break;
      case "rrf":
        if (response.qualified_rrf_evidence[0] !== undefined) response.qualified_rrf_evidence[0].trace_id = foreignTrace;
        break;
      case "observation":
        if (response.observations[0] !== undefined) response.observations[0].trace_id = foreignTrace;
        break;
    }
    vi.mocked(diagnoseExpectedDocument).mockResolvedValueOnce(response);
    renderDiagnostic();
    runDiagnostic(configId);

    expect(await screen.findByRole("alert")).toHaveTextContent(/did not match this exact selection/i);
    expect(document.body).not.toHaveTextContent(foreignTrace);
    expect(screen.queryByText("New live expected-document diagnostic · not original run evidence")).not.toBeInTheDocument();
  });

  it("rejects mixed client-computed evidence nested under a live-source observation", async () => {
    const response = expectedDocumentDiagnosticResponse(baselineId, false, { targetAvailable: false });
    const observation = response.observations[0];
    if (observation === undefined) throw new Error("Expected unavailable observation fixture");
    observation.evidence = [{
      label: "mixed_source_attack",
      value: { kind: "warning", code: "namespace_unavailable" },
      origin: "client_computed",
      observed_at: response.observed_at,
      trace_id: response.trace_id,
    }];
    vi.mocked(diagnoseExpectedDocument).mockResolvedValueOnce(response);
    renderDiagnostic();
    runDiagnostic(baselineId);

    expect(await screen.findByRole("alert")).toHaveTextContent(/did not match this exact selection/i);
    expect(document.body).not.toHaveTextContent("mixed source attack");
    expect(screen.queryByText("New live expected-document diagnostic · not original run evidence")).not.toBeInTheDocument();
  });
});
