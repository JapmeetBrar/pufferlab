import type { components, paths } from "./schema";

type HealthResponse =
  paths["/api/v1/health"]["get"]["responses"][200]["content"]["application/json"];
export type CapabilitiesResponse =
  paths["/api/v1/capabilities"]["get"]["responses"][200]["content"]["application/json"];
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
    value === "configuration_required" ||
    value === "not_found" ||
    value === "namespace_not_ready" ||
    value === "provider_error" ||
    value === "rate_limited" ||
    value === "run_conflict" ||
    value === "internal_error"
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isApiErrorDetail(value: unknown): value is ApiErrorDetail {
  if (!isRecord(value)) {
    return false;
  }
  return (
    isApiErrorCode(value.code) &&
    typeof value.message === "string" &&
    typeof value.retryable === "boolean" &&
    typeof value.trace_id === "string"
  );
}

export async function readApiError(response: Response): Promise<ApiRequestError> {
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

export async function readJsonResponse<ResponseBody>(
  response: Response,
): Promise<ResponseBody> {
  if (!response.ok) {
    throw await readApiError(response);
  }
  return (await response.json()) as ResponseBody;
}

export function apiUrl(path: string): string {
  return `${apiBaseUrl}${path}`;
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(apiUrl("/api/v1/health"), { signal });
  if (!response.ok) {
    throw new Error(`Health request failed with status ${response.status}`);
  }
  return (await response.json()) as HealthResponse;
}

export async function getCapabilities(signal?: AbortSignal): Promise<CapabilitiesResponse> {
  const response = await fetch(apiUrl("/api/v1/capabilities"), { signal });
  return readJsonResponse<CapabilitiesResponse>(response);
}

export async function getRetrievalConfigs(
  signal?: AbortSignal,
): Promise<RetrievalConfigListResponse> {
  const response = await fetch(apiUrl("/api/v1/configs"), { signal });
  return readJsonResponse<RetrievalConfigListResponse>(response);
}

export async function compareSearchConfigs(
  request: SearchCompareRequest,
): Promise<SearchCompareResponse> {
  const response = await fetch(apiUrl("/api/v1/search/compare"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return readJsonResponse<SearchCompareResponse>(response);
}
