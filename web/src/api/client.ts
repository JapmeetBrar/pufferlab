import type { components, paths } from "./schema";

type HealthResponse =
  paths["/api/v1/health"]["get"]["responses"][200]["content"]["application/json"];
export type RetrievalConfigListResponse =
  paths["/api/v1/configs"]["get"]["responses"][200]["content"]["application/json"];
export type SearchCompareRequest =
  paths["/api/v1/search/compare"]["post"]["requestBody"]["content"]["application/json"];
export type SearchCompareResponse =
  paths["/api/v1/search/compare"]["post"]["responses"][200]["content"]["application/json"];
export type ApiErrorDetail = components["schemas"]["ApiErrorDetail"];

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

export class ApiRequestError extends Error {
  constructor(public readonly detail: ApiErrorDetail, public readonly status: number) {
    super(detail.message);
    this.name = "ApiRequestError";
  }
}

function isApiErrorCode(value: unknown): value is ApiErrorDetail["code"] {
  return (
    value === "validation_error" ||
    value === "not_found" ||
    value === "namespace_not_ready" ||
    value === "provider_error" ||
    value === "rate_limited" ||
    value === "run_conflict" ||
    value === "internal_error"
  );
}

function isApiErrorDetail(value: unknown): value is ApiErrorDetail {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    isApiErrorCode(candidate.code) &&
    typeof candidate.message === "string" &&
    typeof candidate.retryable === "boolean" &&
    typeof candidate.trace_id === "string"
  );
}

async function readApiError(response: Response): Promise<ApiRequestError> {
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    payload = undefined;
  }
  if (isApiErrorDetail(payload)) {
    return new ApiRequestError(payload, response.status);
  }
  return new ApiRequestError(
    {
      code: "internal_error",
      message: `Request failed with status ${response.status}`,
      retryable: response.status >= 500,
      trace_id: "unavailable",
    },
    response.status,
  );
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${apiBaseUrl}/api/v1/health`, { signal });
  if (!response.ok) {
    throw new Error(`Health request failed with status ${response.status}`);
  }
  return (await response.json()) as HealthResponse;
}

export async function getRetrievalConfigs(
  signal?: AbortSignal,
): Promise<RetrievalConfigListResponse> {
  const response = await fetch(`${apiBaseUrl}/api/v1/configs`, { signal });
  if (!response.ok) {
    throw await readApiError(response);
  }
  return (await response.json()) as RetrievalConfigListResponse;
}

export async function compareSearchConfigs(
  request: SearchCompareRequest,
): Promise<SearchCompareResponse> {
  const response = await fetch(`${apiBaseUrl}/api/v1/search/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw await readApiError(response);
  }
  return (await response.json()) as SearchCompareResponse;
}
