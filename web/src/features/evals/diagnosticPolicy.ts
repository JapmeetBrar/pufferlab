import type { EvaluationRunQueryDetailResponse } from "../../api/evaluations";

type QueryMode = EvaluationRunQueryDetailResponse["configs"][number]["mode"];

export function diagnosticSubqueryCount(
  mode: QueryMode,
  includeNoFilterCounterfactual: boolean,
): number {
  const storedCount = mode === "bm25" || mode === "vector" ? 2 : 3;
  if (!includeNoFilterCounterfactual) return storedCount;
  return storedCount + (mode === "bm25" || mode === "vector" ? 1 : 2);
}
