import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiRequestError } from "./client";
import {
  cancelEvaluationRun,
  createEvaluationRun,
  getEvaluationRegressions,
  getEvaluationRun,
  listDatasetEvaluationConfigs,
  listEvaluationDatasets,
  listEvaluationQuerySets,
  listEvaluationRuns,
  type CreateEvaluationRunRequest,
} from "./evaluations";

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

function successfulFetch(body: unknown = {}) {
  return vi.fn((...args: [input: string | URL | Request, init?: RequestInit]) => {
    void args;
    return Promise.resolve(jsonResponse(body));
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("evaluation API client", () => {
  it("encodes run and regression URL state and forwards AbortSignal", async () => {
    const fetchMock = successfulFetch();
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();

    await listEvaluationRuns(25, controller.signal);
    await getEvaluationRun("run/id", controller.signal);
    await getEvaluationRegressions(
      "run/id",
      { candidate_config_id: "candidate id", order: "gains", limit: 7 },
      controller.signal,
    );

    expect(requestUrl(fetchMock.mock.calls[0]?.[0] ?? "")).toBe("/api/v1/eval-runs?limit=25");
    expect(fetchMock.mock.calls[0]?.[1]).toEqual({ signal: controller.signal });
    expect(requestUrl(fetchMock.mock.calls[1]?.[0] ?? "")).toBe("/api/v1/eval-runs/run%2Fid");
    expect(requestUrl(fetchMock.mock.calls[2]?.[0] ?? "")).toBe(
      "/api/v1/eval-runs/run%2Fid/regressions?candidate_config_id=candidate+id&order=gains&limit=7",
    );
  });

  it("uses dataset-scoped catalogs without changing the Playground config route", async () => {
    const fetchMock = successfulFetch();
    vi.stubGlobal("fetch", fetchMock);

    await listEvaluationDatasets();
    await listEvaluationQuerySets("dataset/id");
    await listDatasetEvaluationConfigs("dataset/id");

    expect(fetchMock.mock.calls.map(([input]) => requestUrl(input))).toEqual([
      "/api/v1/datasets",
      "/api/v1/query-sets?dataset_version_id=dataset%2Fid",
      "/api/v1/datasets/dataset%2Fid/configs",
    ]);
    expect(fetchMock.mock.calls.some(([input]) => requestUrl(input) === "/api/v1/configs")).toBe(false);
  });

  it("posts the exact generated create body at 202 and exposes no secret or vector field", async () => {
    const response = { contract_version: 1, result: { run: { id: "run-id" } } };
    const fetchMock = vi.fn((...args: [input: string | URL | Request, init?: RequestInit]) => {
      void args;
      return Promise.resolve(jsonResponse(response, 202));
    });
    vi.stubGlobal("fetch", fetchMock);
    const request: CreateEvaluationRunRequest = {
      contract_version: 1,
      query_set_id: "query-set-id",
      baseline_config_id: "baseline-id",
      candidate_config_ids: ["candidate-1", "candidate-2", "candidate-3"],
      random_seed: 20260822,
      max_concurrency: 4,
      warmup_query_count: 5,
    };

    await expect(createEvaluationRun(request)).resolves.toBe(response);
    const init = fetchMock.mock.calls[0]?.[1];
    expect(init).toMatchObject({
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    if (typeof init?.body !== "string") throw new Error("Expected a JSON request body");
    const body = JSON.parse(init.body) as Record<string, unknown>;
    expect(body).toEqual(request);
    expect(body).not.toHaveProperty("api_key");
    expect(body).not.toHaveProperty("namespace");
    expect(body).not.toHaveProperty("vector");
  });

  it("uses a body-free cancel request", async () => {
    const fetchMock = successfulFetch();
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();

    await cancelEvaluationRun("run/id", controller.signal);

    expect(fetchMock).toHaveBeenCalledWith("/api/v1/eval-runs/run%2Fid/cancel", {
      method: "POST",
      signal: controller.signal,
    });
    expect(fetchMock.mock.calls[0]?.[1]).not.toHaveProperty("body");
  });

  it("parses a direct redacted ApiErrorDetail without a nested detail shape", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          jsonResponse(
            {
              code: "run_conflict",
              message: "An equivalent run is already active.",
              retryable: false,
              trace_id: "safe-trace",
              details: { operation: "create_eval_run" },
            },
            409,
          ),
        ),
      ),
    );

    await expect(listEvaluationRuns()).rejects.toEqual(
      new ApiRequestError(
        {
          code: "run_conflict",
          message: "An equivalent run is already active.",
          retryable: false,
          trace_id: "safe-trace",
          details: { operation: "create_eval_run" },
        },
        409,
      ),
    );
  });

  it("falls back safely when an error body is malformed", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse({ detail: "unsafe" }, 503))));

    await expect(getEvaluationRun("run-id")).rejects.toMatchObject({
      status: 503,
      detail: {
        code: "internal_error",
        message: "Request failed with status 503",
        retryable: true,
        trace_id: "unavailable",
      },
    });
  });
});
