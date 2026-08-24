import { describe, expect, it } from "vitest";

import {
  baselineId,
  candidateIds,
  expectedDocumentDiagnosticResponse,
} from "./evalFixtures";

describe("expected-document diagnostic fixtures", () => {
  it.each([
    [baselineId, 2, 3],
    [candidateIds[0], 2, 3],
    [candidateIds[1], 3, 5],
    [candidateIds[2], 3, 5],
  ] as const)("authors exact role/count and scope-bound filter facts for %s", (configId, normalCount, noFilterCount) => {
    const normal = expectedDocumentDiagnosticResponse(configId, false, {
      storedFilterResult: "matched",
    });
    const counterfactual = expectedDocumentDiagnosticResponse(configId, true, {
      storedFilterResult: "matched",
    });

    expect(normal.subqueries).toHaveLength(normalCount);
    expect(counterfactual.subqueries).toHaveLength(noFilterCount);
    expect(normal.subqueries.map((item) => item.ordinal)).toEqual(
      Array.from({ length: normalCount }, (_, index) => index),
    );
    expect(counterfactual.candidate_evidence
      .filter((item) => item.scope === "stored_query")
      .every((item) => item.stored_filter_result === "matched")).toBe(true);
    expect(counterfactual.candidate_evidence
      .filter((item) => item.scope === "no_filter_counterfactual")
      .every((item) => item.stored_filter_result === null && item.certainty === "counterfactual"))
      .toBe(true);
    expect(counterfactual.qualified_rrf_evidence
      .filter((item) => item.scope === "no_filter_counterfactual")
      .every((item) => item.stored_filter_result === null && item.certainty === "counterfactual"))
      .toBe(true);
  });

  it("suppresses downstream claims when the exact target is unavailable", () => {
    const unavailable = expectedDocumentDiagnosticResponse(baselineId, false, {
      targetAvailable: false,
    });

    expect(unavailable.stored_filter_result).toBeNull();
    expect(unavailable.filter_evidence).toEqual([]);
    expect(unavailable.candidate_evidence).toEqual([]);
    expect(unavailable.qualified_rrf_evidence).toEqual([]);
    expect(unavailable.observations[0]?.code).toBe("not_observable");
  });

  it("rejects impossible authored target/filter and hybrid filter-miss combinations", () => {
    expect(() => expectedDocumentDiagnosticResponse(baselineId, false, {
      targetAvailable: false,
      storedFilterResult: "matched",
    })).toThrow(/cannot retain filter evidence/);
    expect(() => expectedDocumentDiagnosticResponse(candidateIds[1], false, {
      storedFilterResult: "not_matched",
    })).toThrow(/require explicit qualified-RRF facts/);
  });
});
