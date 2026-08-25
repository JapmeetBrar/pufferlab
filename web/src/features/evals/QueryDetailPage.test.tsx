import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../api/evaluations", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/evaluations")>();
  return {
    ...actual,
    diagnoseExpectedDocument: vi.fn(),
    getEvaluationRunQuery: vi.fn(),
    replayEvaluationRunQuery: vi.fn(),
  };
});

import { ApiRequestError } from "../../api/client";
import {
  diagnoseExpectedDocument,
  getEvaluationRunQuery,
  replayEvaluationRunQuery,
} from "../../api/evaluations";
import { useAppLocation } from "../../app/routing";
import {
  baselineId,
  candidateIds,
  diagnosticTrace,
  documentId,
  expectedDocumentDiagnosticResponse,
  failedProbeTrace,
  probeTrace,
  queryDetail,
  queryId,
  replayResponse,
  runId,
} from "../../test/evalFixtures";
import { QueryDetailPage } from "./QueryDetailPage";

function TestProvider({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function Harness() {
  const location = useAppLocation();
  return (
    <QueryDetailPage
      runId={runId}
      queryId={queryId}
      routeKey={`/runs/${runId}/queries/${queryId}`}
      routeKind="run-query"
      search={location.search}
    />
  );
}

function renderPage(search = `?left=${baselineId}&right=${candidateIds[0]}`) {
  window.history.replaceState(null, "", `/runs/${runId}/queries/${queryId}${search}`);
  return render(<Harness />, { wrapper: TestProvider });
}

beforeEach(() => {
  vi.mocked(getEvaluationRunQuery).mockResolvedValue(queryDetail());
  vi.mocked(replayEvaluationRunQuery).mockResolvedValue(replayResponse);
  vi.mocked(diagnoseExpectedDocument).mockResolvedValue(expectedDocumentDiagnosticResponse());
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  window.history.replaceState(null, "", "/");
  document.body.style.overflow = "";
});

describe("QueryDetailPage", () => {
  it("renders provider-free durable evidence and canonicalizes UUID-only state", async () => {
    renderPage(`?left=invalid&right=invalid&q=licensed&query_text=licensed`);

    expect(screen.getByRole("heading", { name: "Query forensics", level: 1 })).toHaveFocus();
    expect(await screen.findByRole("heading", { name: "authored local query text" })).toBeVisible();
    expect(screen.getByRole("table", { name: "Durable outcomes for the recorded query" })).toBeVisible();
    const judgments = screen.getByRole("table", {
      name: "Judged documents and durable final ranks",
    });
    expect(within(judgments).getByText("Authored relevant document")).toBeVisible();
    expect(within(judgments).getByText("Authored secondary document")).toBeVisible();
    expect(within(judgments).queryByText(documentId)).not.toBeInTheDocument();
    expect(within(judgments).getByText("Highly relevant")).toBeVisible();
    expect(within(judgments).getByText("Relevant")).toBeVisible();
    expect(within(judgments).getByText("Grade 2")).toBeVisible();
    expect(screen.getByText("A safe recorded failure.")).toBeVisible();
    expect(screen.queryByText(/NOT_OBSERVABLE · original stages/i)).not.toBeInTheDocument();
    expect(replayEvaluationRunQuery).not.toHaveBeenCalled();
    expect(diagnoseExpectedDocument).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(window.location.search).toBe(`?left=${baselineId}&right=${candidateIds[0]}`);
    });
    expect(window.location.href).not.toContain("licensed");
  });

  it("keeps synthetic query evidence read-only with zero replay calls", async () => {
    vi.mocked(getEvaluationRunQuery).mockResolvedValue(queryDetail("synthetic_demo"));
    renderPage();

    expect(await screen.findByText(/Synthetic demo · replay disabled/)).toBeVisible();
    expect(screen.queryByRole("button", { name: /run live replay/i })).not.toBeInTheDocument();
    expect(replayEvaluationRunQuery).not.toHaveBeenCalled();
    expect(diagnoseExpectedDocument).not.toHaveBeenCalled();
  });

  it("falls back safely when an older catalog has no stored document title", async () => {
    const detail = queryDetail();
    vi.mocked(getEvaluationRunQuery).mockResolvedValue({
      ...detail,
      judged_documents: detail.judged_documents.map((document, index) => index === 0
        ? { ...document, title: null }
        : document),
    });
    renderPage();

    const judgments = await screen.findByRole("table", {
      name: "Judged documents and durable final ranks",
    });
    expect(within(judgments).getByText("Title unavailable")).toBeVisible();
    expect(within(judgments).getByText(documentId)).toBeVisible();
  });

  it("keeps live replay unavailable when the durable origin policy denies it", async () => {
    vi.mocked(getEvaluationRunQuery).mockResolvedValue({
      ...queryDetail(),
      live_replay_policy_permitted: false,
    });
    renderPage();

    expect(await screen.findByText(/Replay is disabled by origin policy/)).toBeVisible();
    expect(screen.queryByRole("button", { name: /run live replay/i })).not.toBeInTheDocument();
    expect(replayEvaluationRunQuery).not.toHaveBeenCalled();
    expect(diagnoseExpectedDocument).not.toHaveBeenCalled();
  });

  it("keeps drawer open/config/history actions provider-free until explicit diagnostic confirmation", async () => {
    renderPage();
    await screen.findByRole("heading", { name: "Judged documents" });
    fireEvent.click(screen.getAllByRole("button", { name: "Inspect document" })[0]!);
    const drawer = await screen.findByRole("dialog", { name: "Document evidence" });
    expect(within(drawer).getByText("Authored relevant document")).toBeVisible();
    const config = within(drawer).getByLabelText("Configuration");
    expect(config).toHaveValue("");
    expect(within(drawer).getByRole("button", { name: "Run expected-document diagnostic" })).toBeDisabled();
    expect(diagnoseExpectedDocument).not.toHaveBeenCalled();
    const drawerHref = window.location.href;

    fireEvent.change(config, { target: { value: baselineId } });
    expect(within(drawer).getByText(/exactly 2 ordered subqueries/i)).toBeVisible();
    expect(within(drawer).getByLabelText("Include a same-request no-filter counterfactual")).toBeDisabled();
    expect(diagnoseExpectedDocument).not.toHaveBeenCalled();
    expect(window.location.href).toBe(drawerHref);

    fireEvent.click(within(drawer).getByLabelText("I understand this starts cost-bearing provider work."));
    fireEvent.click(within(drawer).getByRole("button", { name: "Run expected-document diagnostic" }));
    expect(await within(drawer).findByText("New live expected-document diagnostic · not original run evidence")).toBeVisible();
    expect(within(drawer).getByText(new RegExp(diagnosticTrace))).toBeVisible();
    expect(diagnoseExpectedDocument).toHaveBeenCalledWith(
      runId,
      queryId,
      documentId,
      { contract_version: 1, config_id: baselineId, include_no_filter_counterfactual: false },
      expect.any(AbortSignal),
    );
    expect(window.location.href).toBe(drawerHref);
  });

  it("preserves stored and replay evidence when the diagnostic fails", async () => {
    vi.mocked(diagnoseExpectedDocument).mockRejectedValueOnce(new ApiRequestError({
      code: "namespace_not_ready",
      message: "The diagnostic namespace is unavailable.",
      retryable: true,
      trace_id: "safe-diagnostic-failure",
    }, 503));
    renderPage();
    await screen.findByText("authored local query text");
    fireEvent.click(screen.getByRole("button", { name: "Run live replay (cost-bearing)" }));
    await screen.findByText("New live replay · not original run evidence");
    fireEvent.click(screen.getAllByRole("button", { name: /Inspect evidence/ })[0]!);
    const drawer = await screen.findByRole("dialog", { name: "Document evidence" });

    fireEvent.change(within(drawer).getByLabelText("Configuration"), {
      target: { value: baselineId },
    });
    fireEvent.click(within(drawer).getByLabelText("I understand this starts cost-bearing provider work."));
    fireEvent.click(within(drawer).getByRole("button", { name: "Run expected-document diagnostic" }));

    const alert = await within(drawer).findByRole("alert");
    expect(within(alert).getByText("The diagnostic namespace is unavailable.")).toBeVisible();
    expect(within(drawer).getByRole("heading", { name: "Stored run evidence" })).toBeVisible();
    expect(within(drawer).getByRole("heading", { name: "New primary replay" })).toBeVisible();
    expect(screen.getByText("New live replay · not original run evidence")).toBeVisible();
    expect(screen.getByRole("table", { name: "Durable outcomes for the recorded query" })).toBeVisible();
    expect(replayEvaluationRunQuery).toHaveBeenCalledTimes(1);
    expect(diagnoseExpectedDocument).toHaveBeenCalledTimes(1);
  });

  it("keeps grade-zero qrels inspectable but diagnostic-ineligible with zero calls", async () => {
    const detail = queryDetail();
    vi.mocked(getEvaluationRunQuery).mockResolvedValue({
      ...detail,
      query: {
        ...detail.query,
        qrels: detail.query.qrels.map((qrel, index) => index === 0
          ? { ...qrel, relevance_grade: 0 }
          : qrel),
      },
    });
    renderPage();
    await screen.findByRole("heading", { name: "Judged documents" });
    fireEvent.click(screen.getAllByRole("button", { name: "Inspect document" })[0]!);
    const drawer = await screen.findByRole("dialog", { name: "Document evidence" });

    expect(within(drawer).getByText(/Judgment: Not relevant \(grade 0\)/)).toBeVisible();
    expect(within(drawer).getByText(/Only a positively judged document is eligible/)).toBeVisible();
    expect(within(drawer).queryByLabelText("Configuration")).not.toBeInTheDocument();
    expect(diagnoseExpectedDocument).not.toHaveBeenCalled();
  });

  it("runs replay only on explicit action, renders separated failed probes, and resets stale evidence", async () => {
    renderPage();
    await screen.findByRole("heading", { name: "Live replay" });
    expect(replayEvaluationRunQuery).not.toHaveBeenCalled();

    fireEvent.click(screen.getByLabelText(/Include separate counterfactual provenance probes/));
    fireEvent.click(screen.getByRole("button", { name: "Run live replay (cost-bearing)" }));

    expect(await screen.findByText("New live replay · not original run evidence")).toBeVisible();
    expect(screen.getByText(new RegExp(failedProbeTrace))).toBeVisible();
    expect(screen.getByText(/separate probe was unavailable/i)).toBeVisible();
    expect(replayEvaluationRunQuery).toHaveBeenCalledTimes(1);

    fireEvent.change(screen.getByLabelText("Left config"), {
      target: { value: candidateIds[1] },
    });
    await waitFor(() => expect(screen.queryByText("New live replay · not original run evidence")).not.toBeInTheDocument());
    expect(new URLSearchParams(window.location.search).get("left")).toBe(candidateIds[1]);
    expect(replayEvaluationRunQuery).toHaveBeenCalledTimes(1);

    window.history.back();
    await waitFor(() => expect(screen.getByLabelText("Left config")).toHaveValue(baselineId));
  });

  it("renders every discriminated evidence kind for the exact target and traps drawer focus", async () => {
    const storedUnavailability: (typeof replayResponse.observations)[number] = {
      code: "not_observable",
      statement: "Stored original-stage duplicate must stay hidden.",
      config_id: baselineId,
      document_id: documentId,
      origin: "stored_run",
      observed_at: null,
      trace_id: null,
      certainty: "insufficient",
      evidence: [{
        label: "original_stage_evidence",
        value: { kind: "warning", code: "original_stage_evidence_unavailable" },
        origin: "stored_run",
        observed_at: null,
        trace_id: null,
      }],
    };
    vi.mocked(replayEvaluationRunQuery).mockResolvedValueOnce({
      ...replayResponse,
      observations: [...replayResponse.observations, storedUnavailability],
    });
    renderPage();
    await screen.findByRole("heading", { name: "Judged documents" });
    const inspectButtons = screen.getAllByRole("button", { name: "Inspect document" });
    const opener = inspectButtons[0];
    if (opener === undefined) throw new Error("Expected an inspect button");
    fireEvent.click(opener);

    const drawer = await screen.findByRole("dialog", { name: "Document evidence" });
    const close = within(drawer).getByRole("button", { name: "Close document evidence" });
    expect(close).toHaveFocus();
    expect(document.body.style.overflow).toBe("hidden");
    expect(within(drawer).getByText(/Judgment: Highly relevant \(grade 2\)/)).toBeVisible();
    expect(within(drawer).getByText(/Recorded final ranks: BM25 1; Vector ANN 2\./)).toBeVisible();
    expect(within(drawer).queryByText(/NOT_OBSERVABLE · original stages/)).not.toBeInTheDocument();

    fireEvent.click(close);
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(opener).toHaveFocus();

    fireEvent.click(screen.getByRole("button", { name: "Run live replay (cost-bearing)" }));
    await screen.findByText("New live replay · not original run evidence");
    fireEvent.click(screen.getAllByRole("button", { name: /Inspect evidence/ })[0]!);

    const replayDrawer = await screen.findByRole("dialog", { name: "Document evidence" });
    expect(within(replayDrawer).getByText(/Rank 3 at vector candidates/)).toBeVisible();
    expect(within(replayDrawer).getByText(/vector candidates score/)).toBeVisible();
    expect(within(replayDrawer).getAllByText(/20 vector candidates/)).toHaveLength(2);
    expect(within(replayDrawer).getByText(/Present at vector candidates/)).toBeVisible();
    expect(within(replayDrawer).getByText(/Filter field source · matched/)).toBeVisible();
    expect(within(replayDrawer).getByText(/provenance snapshot differs/)).toBeVisible();
    expect(within(replayDrawer).getByText(/1 \/ \(60 \+ 3\)/)).toBeVisible();
    expect(within(replayDrawer).getAllByText(new RegExp(probeTrace)).length).toBeGreaterThan(0);
    expect(within(replayDrawer).getAllByText(/probe unavailable/i).length).toBeGreaterThan(0);
    expect(within(replayDrawer).queryByText("Stored original-stage duplicate must stay hidden.")).not.toBeInTheDocument();
    expect(within(replayDrawer).queryByText(/original stage evidence unavailable/i)).not.toBeInTheDocument();

    const replayClose = within(replayDrawer).getByRole("button", { name: "Close document evidence" });
    const diagnosticConfig = within(replayDrawer).getByLabelText("Configuration");
    fireEvent.keyDown(replayDrawer, { key: "Tab" });
    expect(replayClose).toHaveFocus();
    fireEvent.keyDown(replayDrawer, { key: "Tab", shiftKey: true });
    expect(diagnosticConfig).toHaveFocus();
    fireEvent.keyDown(replayDrawer, { key: "Tab" });
    expect(replayClose).toHaveFocus();
    fireEvent.keyDown(replayDrawer, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(document.body.style.overflow).toBe("");
  });

  it("keeps stored evidence visible when namespace replay fails and retries explicitly", async () => {
    vi.mocked(replayEvaluationRunQuery)
      .mockRejectedValueOnce(
        new ApiRequestError(
          {
            code: "namespace_not_ready",
            message: "The bound namespace is unavailable.",
            retryable: true,
            trace_id: "safe-namespace-trace",
          },
          503,
        ),
      )
      .mockResolvedValueOnce(replayResponse);
    renderPage();
    await screen.findByText("authored local query text");

    fireEvent.click(screen.getByRole("button", { name: "Run live replay (cost-bearing)" }));
    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText(/NOT_OBSERVABLE · provider namespace unavailable/)).toBeVisible();
    expect(screen.getByRole("table", { name: "Durable outcomes for the recorded query" })).toBeVisible();
    fireEvent.click(within(alert).getByRole("button", { name: "Retry explicit replay" }));
    expect(await screen.findByText("New live replay · not original run evidence")).toBeVisible();
    expect(replayEvaluationRunQuery).toHaveBeenCalledTimes(2);
  });

  it("announces and aborts an in-flight explicit replay on unmount", async () => {
    let replaySignal: AbortSignal | undefined;
    vi.mocked(replayEvaluationRunQuery).mockImplementation((_run, _query, _request, signal) => {
      replaySignal = signal;
      return new Promise(() => undefined);
    });
    const view = renderPage();
    await screen.findByText("authored local query text");

    fireEvent.click(screen.getByRole("button", { name: "Run live replay (cost-bearing)" }));
    expect(await screen.findByRole("button", { name: "Running live replay…" })).toBeDisabled();
    expect(screen.getByText("Live replay is loading.")).toBeInTheDocument();
    await waitFor(() => expect(replaySignal).toBeDefined());

    view.unmount();
    expect(replaySignal?.aborted).toBe(true);
  });

  it("distinguishes query not-found from a retryable read failure and aborts on unmount", async () => {
    vi.mocked(getEvaluationRunQuery).mockRejectedValueOnce(
      new ApiRequestError(
        { code: "not_found", message: "Query not found.", retryable: false, trace_id: "safe-404" },
        404,
      ),
    );
    const first = renderPage();
    expect(await screen.findByRole("heading", { name: "Query not found", level: 1 })).toBeVisible();
    first.unmount();

    vi.mocked(getEvaluationRunQuery)
      .mockRejectedValueOnce(
        new ApiRequestError(
          { code: "internal_error", message: "Temporary read failure.", retryable: true, trace_id: "safe-503" },
          503,
        ),
      )
      .mockResolvedValueOnce(queryDetail());
    renderPage();
    const alert = await screen.findByRole("alert");
    fireEvent.click(within(alert).getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("authored local query text")).toBeVisible();

    cleanup();
    let signal: AbortSignal | undefined;
    vi.mocked(getEvaluationRunQuery).mockImplementation((_run, _query, requestSignal) => {
      signal = requestSignal;
      return new Promise(() => undefined);
    });
    const pending = renderPage();
    expect(screen.getByRole("status")).toHaveTextContent("Loading recorded query");
    await waitFor(() => expect(signal).toBeDefined());
    pending.unmount();
    expect(signal?.aborted).toBe(true);
  });

  it("does not add unsupported provider explanations to browser copy", async () => {
    renderPage();
    await screen.findByText("authored local query text");
    const body = document.body.textContent?.toLowerCase() ?? "";
    expect(body).not.toContain("cache was cold");
    expect(body).not.toContain("filter ran before ann");
    expect(body).not.toContain("searched cluster");
  });
});
