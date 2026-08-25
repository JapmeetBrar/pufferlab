import { describe, expect, it } from "vitest";

import { configLabel } from "./configLabels";

describe("configLabel", () => {
  it.each([
    ["bm25", "BM25"],
    ["vector", "Vector ANN"],
    ["hybrid_rrf", "Hybrid RRF"],
    ["hybrid_rerank", "Hybrid + local cross-encoder"],
  ] as const)("presents %s without persisted model-name noise", (mode, expected) => {
    expect(configLabel({ mode })).toBe(expected);
  });
});
