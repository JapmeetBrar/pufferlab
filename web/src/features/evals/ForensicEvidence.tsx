import type { EvaluationRunQueryReplayResponse } from "../../api/evaluations";
import { formatDate } from "./formatters";

type Observation = EvaluationRunQueryReplayResponse["observations"][number];
type EvidenceValue = Observation["evidence"][number]["value"];

const originLabels: Record<Observation["origin"], string> = {
  stored_run: "Stored run",
  live_replay_primary: "Live replay primary",
  live_replay_counterfactual_probe: "Separate counterfactual probe",
  live_expected_document_diagnostic: "Live expected-document diagnostic",
  client_computed: "Client-computed from returned inputs",
};

function evidenceDate(value: string | null): string {
  return value === null ? "Unavailable" : formatDate(value);
}

function scoreText(score: Extract<EvidenceValue, { kind: "score" }>["score"]): string {
  return `${score.value.toLocaleString(undefined, { maximumFractionDigits: 6 })} · ${score.kind.replaceAll("_", " ")}`;
}

export function EvidenceValueView({ value }: { value: EvidenceValue }) {
  switch (value.kind) {
    case "rank":
      return <span>Rank {value.rank} at {value.stage.replaceAll("_", " ")}</span>;
    case "score":
      return <span>{value.stage.replaceAll("_", " ")} score · {scoreText(value.score)}</span>;
    case "candidate_count":
      return <span>{value.count} {value.stage.replaceAll("_", " ")} candidates</span>;
    case "presence":
      return <span>{value.present ? "Present" : "Absent"} at {value.stage.replaceAll("_", " ")}</span>;
    case "filter_result":
      return <span>Filter field {value.field} · {value.matched ? "matched" : "did not match"}</span>;
    case "rrf_contribution":
      return (
        <span>
          {value.stage.replaceAll("_", " ")} · {value.weight} / ({value.rank_constant} + {value.rank})
          {" = "}{value.contribution.toLocaleString(undefined, { maximumFractionDigits: 8 })}
        </span>
      );
    case "warning":
      return <span>{value.code.replaceAll("_", " ")}</span>;
    case "diagnostic_direct_score":
      return <span>{value.signal.toUpperCase()} direct score · {scoreText(value.score)}</span>;
    case "diagnostic_filter_result":
      return (
        <span>
          Filter predicate {value.predicate_ordinal} · path {value.predicate_path.join(".")} · field {value.field}
          {" · "}{value.operator.replaceAll("_", " ")} · {value.result.replaceAll("_", " ")}
        </span>
      );
    case "diagnostic_cutoff_relation":
      return (
        <span>
          {value.scope.replaceAll("_", " ")} · {value.signal.toUpperCase()} · {value.relation.replaceAll("_", " ")}
        </span>
      );
    default: {
      const exhaustive: never = value;
      return exhaustive;
    }
  }
}

export function ForensicEvidence({ observations }: { observations: Observation[] }) {
  if (observations.length === 0) {
    return (
      <div className="forensic-empty" role="note">
        <strong>NOT_OBSERVABLE</strong>
        <span>No typed replay observation was returned for this exact configuration and document.</span>
      </div>
    );
  }
  return (
    <ol className="observation-list">
      {observations.map((observation, index) => (
        <li key={`${observation.origin}-${observation.trace_id ?? "stored"}-${index}`}>
          <div className="observation-heading">
            <strong>{observation.code === "not_observable" ? "NOT_OBSERVABLE" : observation.code.replaceAll("_", " ")}</strong>
            <span>{observation.certainty} · {originLabels[observation.origin]}</span>
          </div>
          <p>{observation.statement}</p>
          <dl className="observation-source">
            <div><dt>Observed</dt><dd>{evidenceDate(observation.observed_at)}</dd></div>
            <div><dt>Trace</dt><dd>{observation.trace_id ?? "None — stored unavailability"}</dd></div>
          </dl>
          {observation.evidence.length > 0 && (
            <ul className="evidence-item-list">
              {observation.evidence.map((item) => (
                <li key={item.label}>
                  <strong>{item.label.replaceAll("_", " ")}</strong>
                  <EvidenceValueView value={item.value} />
                  <small>{originLabels[item.origin]} · {evidenceDate(item.observed_at)} · trace {item.trace_id ?? "none"}</small>
                </li>
              ))}
            </ul>
          )}
        </li>
      ))}
    </ol>
  );
}
