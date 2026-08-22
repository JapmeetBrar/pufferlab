import type { paths } from "./schema";

type HealthResponse =
  paths["/api/v1/health"]["get"]["responses"][200]["content"]["application/json"];

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${apiBaseUrl}/api/v1/health`, { signal });
  if (!response.ok) {
    throw new Error(`Health request failed with status ${response.status}`);
  }
  return (await response.json()) as HealthResponse;
}
