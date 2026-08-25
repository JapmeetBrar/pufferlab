import type { components } from "../api/schema";

type RetrievalConfigSummary = components["schemas"]["RetrievalConfigSummary"];
type RetrievalMode = RetrievalConfigSummary["mode"];

const configLabels: Record<RetrievalMode, string> = {
  bm25: "BM25",
  vector: "Vector ANN",
  hybrid_rrf: "Hybrid RRF",
  hybrid_rerank: "Hybrid + local cross-encoder",
};

export function configLabel(config: Pick<RetrievalConfigSummary, "mode">): string {
  return configLabels[config.mode];
}
