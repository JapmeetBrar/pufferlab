import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../api/evaluations", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../api/evaluations")>();
  return {
    ...actual,
    createEvaluationRun: vi.fn(),
    listDatasetEvaluationConfigs: vi.fn(),
    listEvaluationDatasets: vi.fn(),
    listEvaluationQuerySets: vi.fn(),
    listEvaluationRuns: vi.fn(),
  };
});

import {
  createEvaluationRun,
  listDatasetEvaluationConfigs,
  listEvaluationDatasets,
  listEvaluationQuerySets,
  listEvaluationRuns,
} from "../../api/evaluations";
import {
  configCatalog,
  datasetId,
  liveConfigCatalog,
  liveDatasets,
  liveQuerySets,
  makeRunView,
  querySetId,
  querySets,
  syntheticDatasets,
} from "../../test/evalFixtures";
import { RunListPage } from "./RunListPage";

function TestProvider({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function renderPage() {
  return render(<RunListPage routeKey="/runs" />, { wrapper: TestProvider });
}

beforeEach(() => {
  vi.mocked(listEvaluationRuns).mockResolvedValue({ contract_version: 1, runs: [] });
  vi.mocked(listEvaluationDatasets).mockResolvedValue(syntheticDatasets);
  vi.mocked(listEvaluationQuerySets).mockResolvedValue(querySets);
  vi.mocked(listDatasetEvaluationConfigs).mockResolvedValue(configCatalog);
  vi.mocked(createEvaluationRun).mockReset();
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  window.history.replaceState(null, "", "/");
});

describe("RunListPage", () => {
  it("covers honest loading and empty states", async () => {
    let resolveRuns: ((value: { contract_version: 1; runs: [] }) => void) | undefined;
    vi.mocked(listEvaluationRuns).mockReturnValue(
      new Promise((resolve) => {
        resolveRuns = resolve;
      }),
    );

    renderPage();

    expect(screen.getByRole("heading", { name: "Evaluation runs", level: 1 })).toHaveFocus();
    expect(screen.getByText("Loading evaluation runs…")).toBeVisible();
    resolveRuns?.({ contract_version: 1, runs: [] });
    expect(await screen.findByRole("heading", { name: "No evaluation runs yet" })).toBeVisible();
  });

  it("shows a direct error and retries the run list", async () => {
    vi.mocked(listEvaluationRuns)
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({ contract_version: 1, runs: [] });

    renderPage();

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByRole("heading", { name: "Evaluation runs are unavailable." })).toBeVisible();
    fireEvent.click(within(alert).getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("heading", { name: "No evaluation runs yet" })).toBeVisible();
    expect(listEvaluationRuns).toHaveBeenCalledTimes(2);
  });

  it("renders all six lifecycle states as visible text in a semantic table", async () => {
    const statuses = ["queued", "running", "completed", "failed", "cancelled", "interrupted"] as const;
    vi.mocked(listEvaluationRuns).mockResolvedValue({
      contract_version: 1,
      runs: statuses.map((status, index) =>
        makeRunView(status, { id: `30000000-0000-4000-8000-00000000000${index}` }),
      ),
    });

    renderPage();

    const table = await screen.findByRole("table", { name: "Persisted evaluation runs" });
    expect(within(table).getAllByRole("row")).toHaveLength(7);
    for (const label of ["Queued", "Running", "Completed", "Failed", "Cancelled", "Interrupted"]) {
      expect(within(table).getByText(label)).toBeVisible();
    }
    expect(within(table).getAllByText(/durable attempts/)).toHaveLength(6);
  });

  it("labels synthetic data and prevents every create mutation", async () => {
    vi.mocked(listEvaluationRuns).mockResolvedValue({
      contract_version: 1,
      runs: [makeRunView("completed", { synthetic: true })],
    });

    renderPage();

    expect(await screen.findAllByText("Synthetic demo")).not.toHaveLength(0);
    expect(screen.getByText(/authored offline dataset has no provider timing/i)).toBeVisible();
    const start = screen.getByRole("button", { name: "Start evaluation run" });
    expect(start).toBeDisabled();
    fireEvent.click(start);
    expect(createEvaluationRun).not.toHaveBeenCalled();
  });

  it("creates the canonical live request from generated catalog data and navigates to it", async () => {
    const created = makeRunView("queued", {
      id: "a0000000-0000-4000-8000-00000000000a",
      completedQueries: 0,
    });
    vi.mocked(listEvaluationDatasets).mockResolvedValue(liveDatasets);
    vi.mocked(listEvaluationQuerySets).mockResolvedValue(liveQuerySets);
    vi.mocked(listDatasetEvaluationConfigs).mockResolvedValue(liveConfigCatalog);
    vi.mocked(createEvaluationRun).mockResolvedValue({ contract_version: 1, result: created });

    renderPage();

    const button = await screen.findByRole("button", { name: "Start evaluation run" });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);

    await waitFor(() => expect(createEvaluationRun).toHaveBeenCalledTimes(1));
    expect(createEvaluationRun).toHaveBeenCalledWith({
      contract_version: 1,
      query_set_id: querySetId,
      baseline_config_id: liveConfigCatalog.configs[0]?.id,
      candidate_config_ids: liveConfigCatalog.configs.slice(1).map((config) => config.id),
      random_seed: 20260822,
      max_concurrency: 4,
      warmup_query_count: 5,
    });
    expect(window.location.pathname).toBe(`/runs/${created.run.id}`);
    expect(listEvaluationQuerySets).toHaveBeenCalledWith(datasetId, expect.any(AbortSignal));
  });
});
