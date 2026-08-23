import type { paths } from "./schema";
import { apiUrl, readJsonResponse } from "./client";

type JsonResponse<
  Path extends keyof paths,
  Method extends keyof paths[Path],
  Status extends keyof paths[Path][Method]["responses" & keyof paths[Path][Method]],
> = paths[Path][Method]["responses" & keyof paths[Path][Method]][Status] extends {
  content: { "application/json": infer Body };
}
  ? Body
  : never;

export type EvaluationRunListResponse = JsonResponse<"/api/v1/eval-runs", "get", 200>;
export type EvaluationRunDetailResponse = JsonResponse<"/api/v1/eval-runs/{run_id}", "get", 200>;
export type RegressionResponse = JsonResponse<
  "/api/v1/eval-runs/{run_id}/regressions",
  "get",
  200
>;
export type EvaluationDatasetListResponse = JsonResponse<"/api/v1/datasets", "get", 200>;
export type EvaluationQuerySetListResponse = JsonResponse<"/api/v1/query-sets", "get", 200>;
export type EvaluationConfigCatalogResponse = JsonResponse<
  "/api/v1/datasets/{dataset_version_id}/configs",
  "get",
  200
>;
export type CreateEvaluationRunRequest =
  paths["/api/v1/eval-runs"]["post"]["requestBody"]["content"]["application/json"];
export type CreateEvaluationRunResponse = JsonResponse<"/api/v1/eval-runs", "post", 202>;
export type CancelEvaluationRunResponse = JsonResponse<
  "/api/v1/eval-runs/{run_id}/cancel",
  "post",
  200
>;
export type EvaluationRunQueryDetailResponse = JsonResponse<
  "/api/v1/eval-runs/{run_id}/queries/{query_id}",
  "get",
  200
>;
export type EvaluationRunQueryReplayRequest =
  paths["/api/v1/eval-runs/{run_id}/queries/{query_id}/replay"]["post"]["requestBody"]["content"]["application/json"];
export type EvaluationRunQueryReplayResponse = JsonResponse<
  "/api/v1/eval-runs/{run_id}/queries/{query_id}/replay",
  "post",
  200
>;
export type RegressionQuery =
  paths["/api/v1/eval-runs/{run_id}/regressions"]["get"]["parameters"]["query"];

function encoded(value: string): string {
  return encodeURIComponent(value);
}

export async function listEvaluationRuns(
  limit = 50,
  signal?: AbortSignal,
): Promise<EvaluationRunListResponse> {
  const query = new URLSearchParams({ limit: String(limit) });
  const response = await fetch(apiUrl(`/api/v1/eval-runs?${query.toString()}`), { signal });
  return readJsonResponse<EvaluationRunListResponse>(response);
}

export async function getEvaluationRun(
  runId: string,
  signal?: AbortSignal,
): Promise<EvaluationRunDetailResponse> {
  const response = await fetch(apiUrl(`/api/v1/eval-runs/${encoded(runId)}`), { signal });
  return readJsonResponse<EvaluationRunDetailResponse>(response);
}

export async function getEvaluationRegressions(
  runId: string,
  query: RegressionQuery,
  signal?: AbortSignal,
): Promise<RegressionResponse> {
  const parameters = new URLSearchParams({
    candidate_config_id: query.candidate_config_id,
  });
  if (query.order !== undefined) parameters.set("order", query.order);
  if (query.limit !== undefined) parameters.set("limit", String(query.limit));
  const response = await fetch(
    apiUrl(`/api/v1/eval-runs/${encoded(runId)}/regressions?${parameters.toString()}`),
    { signal },
  );
  return readJsonResponse<RegressionResponse>(response);
}

export async function listEvaluationDatasets(
  signal?: AbortSignal,
): Promise<EvaluationDatasetListResponse> {
  const response = await fetch(apiUrl("/api/v1/datasets"), { signal });
  return readJsonResponse<EvaluationDatasetListResponse>(response);
}

export async function listEvaluationQuerySets(
  datasetVersionId: string,
  signal?: AbortSignal,
): Promise<EvaluationQuerySetListResponse> {
  const query = new URLSearchParams({ dataset_version_id: datasetVersionId });
  const response = await fetch(apiUrl(`/api/v1/query-sets?${query.toString()}`), { signal });
  return readJsonResponse<EvaluationQuerySetListResponse>(response);
}

export async function listDatasetEvaluationConfigs(
  datasetVersionId: string,
  signal?: AbortSignal,
): Promise<EvaluationConfigCatalogResponse> {
  const response = await fetch(
    apiUrl(`/api/v1/datasets/${encoded(datasetVersionId)}/configs`),
    { signal },
  );
  return readJsonResponse<EvaluationConfigCatalogResponse>(response);
}

export async function createEvaluationRun(
  request: CreateEvaluationRunRequest,
  signal?: AbortSignal,
): Promise<CreateEvaluationRunResponse> {
  const response = await fetch(apiUrl("/api/v1/eval-runs"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  return readJsonResponse<CreateEvaluationRunResponse>(response);
}

export async function cancelEvaluationRun(
  runId: string,
  signal?: AbortSignal,
): Promise<CancelEvaluationRunResponse> {
  const response = await fetch(apiUrl(`/api/v1/eval-runs/${encoded(runId)}/cancel`), {
    method: "POST",
    signal,
  });
  return readJsonResponse<CancelEvaluationRunResponse>(response);
}

export async function getEvaluationRunQuery(
  runId: string,
  queryId: string,
  signal?: AbortSignal,
): Promise<EvaluationRunQueryDetailResponse> {
  const response = await fetch(
    apiUrl(`/api/v1/eval-runs/${encoded(runId)}/queries/${encoded(queryId)}`),
    { signal },
  );
  return readJsonResponse<EvaluationRunQueryDetailResponse>(response);
}

export async function replayEvaluationRunQuery(
  runId: string,
  queryId: string,
  request: EvaluationRunQueryReplayRequest,
  signal?: AbortSignal,
): Promise<EvaluationRunQueryReplayResponse> {
  const response = await fetch(
    apiUrl(`/api/v1/eval-runs/${encoded(runId)}/queries/${encoded(queryId)}/replay`),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      signal,
    },
  );
  return readJsonResponse<EvaluationRunQueryReplayResponse>(response);
}
