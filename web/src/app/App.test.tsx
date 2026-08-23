import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  RetrievalConfigListResponse,
  SearchCompareResponse,
} from "../api/client";
import {
  baselineId,
  candidateIds,
  documentId,
  makeRunView,
  queryDetail,
  queryId,
  regressionResponse,
  runDetail,
  runId,
} from "../test/evalFixtures";
import { App } from "./App";

const leftId = "11111111-1111-4111-8111-111111111111";
const rightId = "22222222-2222-4222-8222-222222222222";
type Config = RetrievalConfigListResponse["configs"][number];

const bm25Config: Config = {
  id: leftId,
  name: "Exact terms",
  mode: "bm25",
  revision: 1,
  config_hash: "bm25-hash",
};
const vectorConfig: Config = {
  id: rightId,
  name: "Semantic neighbors",
  mode: "vector",
  revision: 1,
  config_hash: "vector-hash",
};

const configs: RetrievalConfigListResponse = {
  contract_version: 1,
  configs: [bm25Config, vectorConfig],
};

const comparison: SearchCompareResponse = {
  contract_version: 1,
  query_id: null,
  query_text: "permission mode",
  observability_notice: "These statements describe observed candidates and scores, not provider internals.",
  overlap: [
    {
      left_config_id: leftId,
      right_config_id: rightId,
      intersection_count: 1,
      left_count: 1,
      right_count: 1,
      jaccard: 1,
    },
  ],
  rank_movements: [
    {
      document_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      ranks_by_config: { [leftId]: 1, [rightId]: 2 },
      max_absolute_delta: 1,
    },
  ],
  results: [
    {
      config: bm25Config,
      trace_id: "33333333-3333-4333-8333-333333333333",
      candidate_counts: { bm25_candidates: 1 },
      timings: [
        { stage: "turbopuffer", duration_ms: 7.4, measurement: "client_wall_clock" },
        { stage: "provenance_probe", duration_ms: 1.2, measurement: "client_wall_clock" },
      ],
      warnings: [],
      hits: [
        {
          document_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
          external_id: "chmod.1#symbolic-modes",
          title: "chmod — change file modes",
          body_excerpt: "Symbolic modes describe who can read, write, and execute a file.",
          url: "https://man7.org/linux/man-pages/man1/chmod.1.html",
          final_rank: 1,
          relevance_grade: 2,
          final_score: {
            value: 12.75,
            kind: "bm25",
            direction: "higher_is_better",
            source: "turbopuffer_dist",
          },
          highlights: [],
          stage_membership: [
            {
              stage: "final",
              rank: 1,
              score: {
                value: 12.75,
                kind: "bm25",
                direction: "higher_is_better",
                source: "turbopuffer_dist",
              },
            },
          ],
          attributes: {},
        },
      ],
    },
    {
      config: vectorConfig,
      trace_id: "44444444-4444-4444-8444-444444444444",
      candidate_counts: { vector_candidates: 1 },
      timings: [
        { stage: "embed", duration_ms: 3.1, measurement: "client_wall_clock" },
        { stage: "turbopuffer", duration_ms: 6.2, measurement: "client_wall_clock" },
      ],
      warnings: [],
      hits: [
        {
          document_id: "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
          external_id: "find.1#permissions",
          title: "find — select files by permissions",
          body_excerpt: "Match files whose permission bits satisfy the requested mode.",
          url: null,
          final_rank: 1,
          relevance_grade: null,
          final_score: {
            value: 0.14321,
            kind: "vector_distance",
            direction: "lower_is_better",
            source: "turbopuffer_dist",
          },
          highlights: [],
          stage_membership: [],
          attributes: {},
        },
      ],
    },
  ],
};

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

function requestUrl(input: string | URL | Request): string {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.href;
  return input.url;
}

function requestBody(init: RequestInit | undefined): Record<string, unknown> {
  if (typeof init?.body !== "string") throw new Error("Expected a JSON request body");
  return JSON.parse(init.body) as Record<string, unknown>;
}

function defaultFetch(compareBody: SearchCompareResponse = comparison) {
  return vi.fn((...args: [input: string | URL | Request, init?: RequestInit]) => {
    const url = requestUrl(args[0]);
    if (url.endsWith("/api/v1/health")) {
      return Promise.resolve(jsonResponse({ contract_version: 1, status: "ok", version: "0.1.0" }));
    }
    if (url.endsWith("/api/v1/configs")) {
      return Promise.resolve(jsonResponse(configs));
    }
    if (url.endsWith("/api/v1/search/compare")) {
      return Promise.resolve(jsonResponse(compareBody));
    }
    return Promise.reject(new Error(`Unexpected request: ${url}`));
  });
}

function renderApp() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.history.replaceState(null, "", "/");
});

describe("App playground", () => {
  it("loads seeded configs and renders typed BM25 and vector evidence", async () => {
    const fetchMock = defaultFetch();
    vi.stubGlobal("fetch", fetchMock);
    renderApp();

    expect(screen.getByRole("heading", { name: "One query. Two retrieval instincts." })).toBeVisible();
    expect(screen.getByRole("button", { name: "Compare results" })).toBeDisabled();
    expect(await screen.findByText("API 0.1.0 ready")).toBeVisible();
    expect(await screen.findAllByRole("option", { name: "Exact terms · bm25" })).toHaveLength(2);

    fireEvent.change(screen.getByLabelText("Search query"), { target: { value: "permission mode" } });
    fireEvent.click(screen.getByRole("button", { name: "Compare results" }));

    expect(await screen.findByRole("heading", { name: "Results for “permission mode”" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Exact terms" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Semantic neighbors" })).toBeVisible();
    expect(screen.getByText("chmod.1#symbolic-modes")).toBeVisible();
    expect(screen.getByText("find.1#permissions")).toBeVisible();
    expect(screen.getByText("12.75 · bm25 · higher is better")).toBeVisible();
    expect(screen.getByText("0.14321 · vector distance · lower is better")).toBeVisible();
    expect(screen.getByLabelText("Exact terms request timings")).toHaveTextContent(
      "Provider7.4 ms client wall clock",
    );
    expect(screen.getByText(/Debug provenance probe · 1.2 ms client wall clock · measured separately/)).toBeVisible();
    expect(screen.getByText(/observed candidates and scores, not provider internals/)).toBeVisible();
    expect(screen.getAllByLabelText("Rank 1")).toHaveLength(2);

    const source = screen.getByRole("link", { name: "Open source for chmod — change file modes" });
    expect(source).toHaveAttribute("href", "https://man7.org/linux/man-pages/man1/chmod.1.html");

    const postCall = fetchMock.mock.calls.find(([input]) => requestUrl(input).endsWith("/api/v1/search/compare"));
    if (postCall === undefined) throw new Error("Expected comparison POST");
    const init = postCall[1];
    const posted = requestBody(init);
    expect(posted).toEqual({
      contract_version: 1,
      query_text: "permission mode",
      config_ids: [leftId, rightId],
      debug_provenance: true,
    });
    expect(posted).not.toHaveProperty("api_key");
    expect(posted).not.toHaveProperty("vector");
    expect(init?.headers).toEqual({ "Content-Type": "application/json" });
  });

  it("restores the query and both configs from a stable URL", async () => {
    window.history.replaceState(null, "", `/?q=restored+query&left=${rightId}&right=${leftId}`);
    const fetchMock = defaultFetch();
    vi.stubGlobal("fetch", fetchMock);
    renderApp();

    expect(screen.getByLabelText("Search query")).toHaveValue("restored query");
    await waitFor(() => expect(screen.getByLabelText("Left result set")).toHaveValue(rightId));
    expect(screen.getByLabelText("Right result set")).toHaveValue(leftId);
    fireEvent.click(screen.getByRole("button", { name: "Compare results" }));
    await screen.findByRole("heading", { name: "Results for “permission mode”" });

    const params = new URLSearchParams(window.location.search);
    expect(params.get("q")).toBe("restored query");
    expect(params.get("left")).toBe(rightId);
    expect(params.get("right")).toBe(leftId);
    const postCall = fetchMock.mock.calls.find(([input]) => requestUrl(input).endsWith("/api/v1/search/compare"));
    if (postCall === undefined) throw new Error("Expected comparison POST");
    const posted = requestBody(postCall[1]);
    expect(posted.config_ids).toEqual([rightId, leftId]);
  });

  it("shows connecting, empty-config, and disabled states honestly", async () => {
    let resolveConfigs: ((value: Response) => void) | undefined;
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const url = requestUrl(input);
      if (url.endsWith("/api/v1/health")) {
        return Promise.resolve(jsonResponse({ contract_version: 1, status: "ok", version: "0.1.0" }));
      }
      if (url.endsWith("/api/v1/configs")) {
        return new Promise<Response>((resolve) => { resolveConfigs = resolve; });
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    renderApp();

    expect(screen.getByText("Loading retrieval configurations…")).toBeVisible();
    expect(screen.getByRole("button", { name: "Compare results" })).toBeDisabled();
    resolveConfigs?.(jsonResponse({ contract_version: 1, configs: [] }));
    expect(await screen.findByText("No retrieval configurations have been seeded yet.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Compare results" })).toBeDisabled();
  });

  it("disables the form and announces an in-flight comparison", async () => {
    let resolveComparison: ((value: Response) => void) | undefined;
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const url = requestUrl(input);
      if (url.endsWith("/api/v1/health")) {
        return Promise.resolve(jsonResponse({ contract_version: 1, status: "ok", version: "0.1.0" }));
      }
      if (url.endsWith("/api/v1/configs")) return Promise.resolve(jsonResponse(configs));
      if (url.endsWith("/api/v1/search/compare")) {
        return new Promise<Response>((resolve) => { resolveComparison = resolve; });
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    renderApp();

    await screen.findAllByRole("option", { name: "Exact terms · bm25" });
    fireEvent.change(screen.getByLabelText("Search query"), { target: { value: "permission mode" } });
    fireEvent.click(screen.getByRole("button", { name: "Compare results" }));
    expect(await screen.findByRole("button", { name: "Comparing…" })).toBeDisabled();
    expect(screen.getByText("Comparison is loading.")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Configurations to compare" })).toBeDisabled();

    resolveComparison?.(jsonResponse(comparison));
    expect(await screen.findByRole("heading", { name: "Results for “permission mode”" })).toBeVisible();
  });

  it("parses a direct API error safely and retries the last comparison", async () => {
    let postCount = 0;
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const url = requestUrl(input);
      if (url.endsWith("/api/v1/health")) {
        return Promise.resolve(jsonResponse({ contract_version: 1, status: "ok", version: "0.1.0" }));
      }
      if (url.endsWith("/api/v1/configs")) return Promise.resolve(jsonResponse(configs));
      if (url.endsWith("/api/v1/search/compare")) {
        postCount += 1;
        return Promise.resolve(
          postCount === 1
            ? jsonResponse({ code: "provider_error", message: "The search provider is unavailable.", retryable: true, trace_id: "safe-trace-42" }, 502)
            : jsonResponse(comparison),
        );
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    renderApp();

    await screen.findAllByRole("option", { name: "Exact terms · bm25" });
    fireEvent.change(screen.getByLabelText("Search query"), { target: { value: "permission mode" } });
    fireEvent.click(screen.getByRole("button", { name: "Compare results" }));
    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText("The search provider is unavailable.")).toBeVisible();
    expect(within(alert).getByText("Trace safe-trace-42")).toBeVisible();
    fireEvent.click(within(alert).getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("heading", { name: "Results for “permission mode”" })).toBeVisible();
    expect(postCount).toBe(2);
  });

  it("supports config-load retry and an empty compare response", async () => {
    let configCount = 0;
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const url = requestUrl(input);
      if (url.endsWith("/api/v1/health")) {
        return Promise.resolve(jsonResponse({ contract_version: 1, status: "ok", version: "0.1.0" }));
      }
      if (url.endsWith("/api/v1/configs")) {
        configCount += 1;
        return Promise.resolve(configCount === 1 ? jsonResponse({ message: "bad gateway" }, 502) : jsonResponse(configs));
      }
      if (url.endsWith("/api/v1/search/compare")) {
        return Promise.resolve(jsonResponse({ ...comparison, results: [], overlap: [], rank_movements: [] }));
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    renderApp();

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText("Retrieval configurations are unavailable.")).toBeVisible();
    fireEvent.click(within(alert).getByRole("button", { name: "Retry" }));
    await screen.findAllByRole("option", { name: "Exact terms · bm25" });
    fireEvent.change(screen.getByLabelText("Search query"), { target: { value: "no matches" } });
    fireEvent.click(screen.getByRole("button", { name: "Compare results" }));
    expect(await screen.findByText("No comparison results were returned.")).toBeVisible();
  });
});

describe("App routing", () => {
  it("opens a frozen forensic Playground link with a provider-free GET and no config fetch", async () => {
    const fetchMock = vi.fn((...args: [input: string | URL | Request, init?: RequestInit]) => {
      const url = requestUrl(args[0]);
      if (url.endsWith("/api/v1/health")) {
        return Promise.resolve(jsonResponse({ contract_version: 1, status: "ok", version: "0.1.0" }));
      }
      if (url.endsWith(`/api/v1/eval-runs/${runId}/queries/${queryId}`)) {
        return Promise.resolve(jsonResponse(queryDetail()));
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    window.history.replaceState(
      null,
      "",
      `/playground?run=${runId}&query=${queryId}&left=${baselineId}&right=${candidateIds[0]}&document=${documentId}`,
    );
    renderApp();

    const heading = await screen.findByRole("heading", { name: "Query forensics", level: 1 });
    expect(heading).toHaveFocus();
    expect(await screen.findByText("authored local query text")).toBeVisible();
    const drawer = await screen.findByRole("dialog", { name: "Document evidence" });
    expect(fetchMock.mock.calls.some(([input]) => requestUrl(input).endsWith("/api/v1/configs"))).toBe(false);
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === "POST")).toBe(false);
    expect(window.location.search).not.toContain("query_text");
    fireEvent.click(within(drawer).getByRole("button", { name: "Close document evidence" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(heading).toHaveFocus();
  });

  it("restores a document drawer through nested-route back, forward, and refresh-safe URL state", async () => {
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const url = requestUrl(input);
      if (url.endsWith("/api/v1/health")) {
        return Promise.resolve(jsonResponse({ contract_version: 1, status: "ok", version: "0.1.0" }));
      }
      if (url.endsWith(`/api/v1/eval-runs/${runId}/queries/${queryId}`)) {
        return Promise.resolve(jsonResponse(queryDetail()));
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    window.history.replaceState(
      null,
      "",
      `/runs/${runId}/queries/${queryId}?left=${baselineId}&right=${candidateIds[0]}`,
    );
    renderApp();

    await screen.findByText("authored local query text");
    const opener = screen.getAllByRole("button", { name: "Inspect document" })[0];
    if (opener === undefined) throw new Error("Expected a document evidence opener");
    fireEvent.click(opener);
    expect(await screen.findByRole("dialog", { name: "Document evidence" })).toBeVisible();
    await waitFor(() => expect(new URLSearchParams(window.location.search).get("document")).toBe(documentId));

    window.history.back();
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(opener).toHaveFocus();
    expect(new URLSearchParams(window.location.search).get("document")).toBeNull();

    window.history.forward();
    const restoredDrawer = await screen.findByRole("dialog", { name: "Document evidence" });
    expect(new URLSearchParams(window.location.search).get("document")).toBe(documentId);
    expect(within(restoredDrawer).getByRole("button", { name: "Close document evidence" })).toHaveFocus();
  });

  it("uses semantic navigation, moves focus, and restores history without reloading", async () => {
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const url = requestUrl(input);
      if (url.endsWith("/api/v1/health")) {
        return Promise.resolve(jsonResponse({ contract_version: 1, status: "ok", version: "0.1.0" }));
      }
      if (url.endsWith("/api/v1/eval-runs?limit=50")) {
        return Promise.resolve(jsonResponse({ contract_version: 1, runs: [] }));
      }
      if (url.endsWith("/api/v1/datasets")) {
        return Promise.resolve(jsonResponse({ contract_version: 1, datasets: [] }));
      }
      if (url.endsWith("/api/v1/configs")) return Promise.resolve(jsonResponse(configs));
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    window.history.replaceState(null, "", "/");
    renderApp();

    const playgroundHeading = screen.getByRole("heading", {
      name: "One query. Two retrieval instincts.",
      level: 1,
    });
    expect(playgroundHeading).toHaveFocus();
    const navigation = screen.getByRole("navigation", { name: "Primary navigation" });
    fireEvent.click(within(navigation).getByRole("link", { name: "Evaluation runs" }));

    const runsHeading = await screen.findByRole("heading", { name: "Evaluation runs", level: 1 });
    expect(runsHeading).toHaveFocus();
    expect(window.location.pathname).toBe("/runs");
    expect(within(navigation).getByRole("link", { name: "Evaluation runs" })).toHaveAttribute(
      "aria-current",
      "page",
    );

    window.history.back();
    await waitFor(() => expect(window.location.pathname).toBe("/"));
    expect(await screen.findByRole("heading", { name: "One query. Two retrieval instincts." })).toHaveFocus();
  });

  it("keeps regression candidate, order, and limit in navigable URL state", async () => {
    const view = makeRunView("completed", { synthetic: true });
    const fetchMock = vi.fn((input: string | URL | Request) => {
      const url = requestUrl(input);
      if (url.endsWith("/api/v1/health")) {
        return Promise.resolve(jsonResponse({ contract_version: 1, status: "ok", version: "0.1.0" }));
      }
      if (url.includes(`/api/v1/eval-runs/${runId}/regressions?`)) {
        const parsed = new URL(url, window.location.origin);
        const order = parsed.searchParams.get("order") === "gains" ? "gains" : "regressions";
        return Promise.resolve(jsonResponse({ ...regressionResponse, order }));
      }
      if (url.endsWith(`/api/v1/eval-runs/${runId}`)) {
        return Promise.resolve(jsonResponse(runDetail(view)));
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    window.history.replaceState(
      null,
      "",
      `/runs/${runId}?candidate=${candidateIds[0]}&order=regressions&limit=10`,
    );
    renderApp();

    expect(await screen.findByRole("heading", { name: "Evaluation run", level: 1 })).toHaveFocus();
    const order = await screen.findByLabelText("Order");
    fireEvent.change(order, { target: { value: "gains" } });
    await waitFor(() => expect(new URLSearchParams(window.location.search).get("order")).toBe("gains"));
    expect(new URLSearchParams(window.location.search).get("candidate")).toBe(candidateIds[0]);
    expect(new URLSearchParams(window.location.search).get("limit")).toBe("10");

    window.history.back();
    await waitFor(() => expect(new URLSearchParams(window.location.search).get("order")).toBe("regressions"));
    expect(screen.getByLabelText("Order")).toHaveValue("regressions");
  });

  it("renders and focuses a route-level not-found state", () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: string | URL | Request) => {
        if (requestUrl(input).endsWith("/api/v1/health")) {
          return Promise.resolve(jsonResponse({ contract_version: 1, status: "ok", version: "0.1.0" }));
        }
        return Promise.reject(new Error("Unexpected request"));
      }),
    );
    window.history.replaceState(null, "", "/missing");
    renderApp();

    expect(screen.getByRole("main")).toHaveAttribute("id", "main-content");
    expect(screen.getByRole("heading", { name: "Page not found", level: 1 })).toHaveFocus();
    expect(screen.getByRole("link", { name: "View evaluation runs" })).toHaveAttribute("href", "/runs");
  });

  it("never fetches mutation routes while displaying a synthetic run", async () => {
    const fetchMock = vi.fn((...args: [input: string | URL | Request, init?: RequestInit]) => {
      const url = requestUrl(args[0]);
      if (url.endsWith("/api/v1/health")) {
        return Promise.resolve(jsonResponse({ contract_version: 1, status: "ok", version: "0.1.0" }));
      }
      if (url.endsWith(`/api/v1/eval-runs/${runId}`)) {
        return Promise.resolve(jsonResponse(runDetail(makeRunView("completed", { synthetic: true }))));
      }
      if (url.includes(`/api/v1/eval-runs/${runId}/regressions?`)) {
        return Promise.resolve(jsonResponse(regressionResponse));
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    window.history.replaceState(null, "", `/runs/${runId}`);
    renderApp();

    expect(await screen.findByText(/create and replay actions are disabled/i)).toBeVisible();
    expect(screen.queryByRole("button", { name: /replay/i })).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === "POST")).toBe(false);
    expect(fetchMock.mock.calls.every(([, init]) => init?.body === undefined)).toBe(true);
  });
});
