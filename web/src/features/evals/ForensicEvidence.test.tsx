import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { EvaluationRunQueryReplayResponse } from "../../api/evaluations";
import { EvidenceValueView, ForensicEvidence } from "./ForensicEvidence";

type Observation = EvaluationRunQueryReplayResponse["observations"][number];
type EvidenceValue = Observation["evidence"][number]["value"];

afterEach(cleanup);

describe("ForensicEvidence shared diagnostic values", () => {
  it("renders the new target-safe shared evidence values exhaustively", () => {
    const values: EvidenceValue[] = [
      {
        kind: "diagnostic_direct_score",
        signal: "bm25",
        score: {
          kind: "bm25",
          value: 3.5,
          direction: "higher_is_better",
          source: "compute_attribute",
        },
      },
      {
        kind: "diagnostic_filter_result",
        predicate_ordinal: 2,
        predicate_path: [1, 0],
        field: "category",
        operator: "eq",
        result: "not_matched",
      },
      {
        kind: "diagnostic_cutoff_relation",
        scope: "no_filter_counterfactual",
        signal: "ann",
        relation: "outside_candidates",
      },
    ];

    const { rerender } = render(<EvidenceValueView value={values[0]!} />);
    expect(screen.getByText(/BM25 direct score/)).toBeVisible();
    rerender(<EvidenceValueView value={values[1]!} />);
    expect(screen.getByText(/Filter predicate 2 · path 1\.0 · field category · eq · not matched/)).toBeVisible();
    rerender(<EvidenceValueView value={values[2]!} />);
    expect(screen.getByText(/no filter counterfactual · ANN · outside candidates/)).toBeVisible();
  });

  it("renders the fixed diagnostic origin label without adding diagnostic UI actions", () => {
    const observation: Observation = {
      code: "not_observable",
      statement: "The target was unavailable in this provider snapshot.",
      certainty: "insufficient",
      origin: "live_expected_document_diagnostic",
      document_id: "00000000-0000-4000-8000-000000000001",
      config_id: "00000000-0000-4000-8000-000000000002",
      observed_at: "2026-08-23T12:00:00Z",
      trace_id: "00000000-0000-4000-8000-000000000003",
      evidence: [],
    };

    render(<ForensicEvidence observations={[observation]} />);

    expect(screen.getByText(/insufficient · Live expected-document diagnostic/)).toBeVisible();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});
