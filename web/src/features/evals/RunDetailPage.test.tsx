import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../api/evaluations", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/evaluations")>();
  return {
    ...actual,
    getEvaluationRegressions: vi.fn(),
    getEvaluationRun: vi.fn(),
  };
});

import { ApiRequestError } from "../../api/client";
import { getEvaluationRegressions, getEvaluationRun } from "../../api/evaluations";
import {
  candidateIds,
  makeRunView,
  regressionResponse,
  runDetail,
  runId,
} from "../../test/evalFixtures";
import { ACTIVE_RUN_POLL_INTERVAL_MS, RunDetailPage } from "./RunDetailPage";

function TestProvider({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function renderPage(search = "") {
  return render(
    <RunDetailPage runId={runId} routeKey={`/runs/${runId}`} search={search} />,
    { wrapper: TestProvider },
  );
}

beforeEach(() => {
  vi.mocked(getEvaluationRun).mockResolvedValue(runDetail());
  vi.mocked(getEvaluationRegressions).mockResolvedValue(regressionResponse);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.useRealTimers();
  window.history.replaceState(null, "", "/");
});

describe("RunDetailPage", () => {
  it("renders synthetic metrics, null timing, complete exclusion coverage, and safe deep links", async () => {
    const synthetic = makeRunView("completed", { synthetic: true });
    vi.mocked(getEvaluationRun).mockResolvedValue(runDetail(synthetic));
    window.history.replaceState(
      null,
      "",
      `/runs/${runId}?candidate=invalid&order=invalid&limit=99&q=licensed-text`,
    );

    renderPage(window.location.search);

    expect(screen.getByRole("heading", { name: "Evaluation run", level: 1 })).toHaveFocus();
    expect(await screen.findByText("Completed", { selector: ".status-badge" })).toBeVisible();
    expect(screen.getByText(/Quality comes from authored judgments and ranks/)).toBeVisible();
    expect(screen.getByText(/not provider service time or a benchmark/)).toBeVisible();
    const metrics = screen.getByRole("table", { name: "Final metrics by retrieval configuration" });
    expect(within(metrics).getAllByText("Unavailable")).toHaveLength(8);
    expect(within(metrics).getAllByText("sample count 0")).toHaveLength(8);
    expect(within(metrics).queryByRole("columnheader", { name: "Error rate" })).not.toBeInTheDocument();

    const coverage = await screen.findByLabelText("Regression coverage");
    expect(screen.queryByRole("columnheader", { name: "Judged rank changes" })).not.toBeInTheDocument();
    for (const label of [
      "Baseline missing",
      "Candidate missing",
      "Baseline failed",
      "Candidate failed",
      "Both failed",
      "No positive judgments",
    ]) {
      expect(within(coverage).getByText(label)).toBeVisible();
    }
    const link = screen.getByRole("link", { name: "Inspect recorded query" });
    const linkUrl = new URL(link.getAttribute("href") ?? "", window.location.origin);
    expect([...linkUrl.searchParams.keys()]).toEqual(["run", "query", "left", "right", "document"]);
    expect(linkUrl.searchParams.has("q")).toBe(false);
    expect(linkUrl.searchParams.has("query_text")).toBe(false);

    await waitFor(() => {
      expect(window.location.search).toBe(
        `?candidate=${candidateIds[0]}&order=regressions&limit=10`,
      );
    });
    expect(getEvaluationRegressions).toHaveBeenCalledWith(
      runId,
      { candidate_config_id: candidateIds[0], order: "regressions", limit: 10 },
      expect.any(AbortSignal),
    );
  });

  it.each(["queued", "running", "completed", "failed", "cancelled", "interrupted"] as const)(
    "renders the %s lifecycle state with durable progress copy",
    async (status) => {
      vi.mocked(getEvaluationRun).mockResolvedValue(runDetail(makeRunView(status)));
      renderPage();

      expect(
        await screen.findByText(status[0]?.toUpperCase() + status.slice(1), {
          selector: ".status-badge",
        }),
      ).toBeVisible();
      expect(screen.getByRole("progressbar", { name: /queries/i })).toBeVisible();
      expect(
        screen.getByText(
          status === "queued" || status === "running"
            ? /Progress refreshes automatically\.$/
            : /Polling stopped\.$/,
        ),
      ).toHaveAttribute("role", "status");
    },
  );

  it("never labels partial summaries as final", async () => {
    vi.mocked(getEvaluationRun).mockResolvedValue(
      runDetail(makeRunView("running", { completedQueries: 12, withSummaries: true })),
    );

    renderPage();

    expect(await screen.findByText("Partial metrics · not final")).toBeVisible();
    expect(screen.queryByText("Final metrics")).not.toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Partial metrics by retrieval configuration" })).toBeVisible();
  });

  it("stops polling as soon as an active run becomes terminal", async () => {
    vi.useFakeTimers();
    vi.mocked(getEvaluationRun)
      .mockResolvedValueOnce(runDetail(makeRunView("running")))
      .mockResolvedValue(runDetail(makeRunView("completed")));

    renderPage();
    await act(async () => { await Promise.resolve(); });
    expect(getEvaluationRun).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACTIVE_RUN_POLL_INTERVAL_MS);
    });
    expect(getEvaluationRun).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(ACTIVE_RUN_POLL_INTERVAL_MS * 3);
    });
    expect(getEvaluationRun).toHaveBeenCalledTimes(2);
  });

  it("aborts the active detail request on unmount", async () => {
    let requestSignal: AbortSignal | undefined;
    vi.mocked(getEvaluationRun).mockImplementation((_id, signal) => {
      requestSignal = signal;
      return new Promise(() => undefined);
    });

    const view = renderPage();
    await waitFor(() => expect(requestSignal).toBeDefined());
    expect(requestSignal?.aborted).toBe(false);
    view.unmount();
    expect(requestSignal?.aborted).toBe(true);
  });

  it("distinguishes not-found from a retryable detail failure", async () => {
    vi.mocked(getEvaluationRun).mockRejectedValueOnce(
      new ApiRequestError(
        { code: "not_found", message: "Run not found.", retryable: false, trace_id: "safe-404" },
        404,
      ),
    );
    const first = renderPage();
    expect(await screen.findByRole("heading", { name: "Run not found", level: 1 })).toBeVisible();
    expect(screen.getByRole("heading", { name: "No run matches this URL" })).toBeVisible();
    first.unmount();

    vi.mocked(getEvaluationRun)
      .mockRejectedValueOnce(
        new ApiRequestError(
          { code: "internal_error", message: "Temporarily unavailable.", retryable: true, trace_id: "safe-503" },
          503,
        ),
      )
      .mockResolvedValueOnce(runDetail());
    renderPage();
    const alert = await screen.findByRole("alert");
    fireEvent.click(within(alert).getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("Completed", { selector: ".status-badge" })).toBeVisible();
    expect(getEvaluationRun).toHaveBeenCalledTimes(3);
  });
});
