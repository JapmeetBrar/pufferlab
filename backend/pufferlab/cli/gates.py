"""Safe deterministic rendering and exits for the provider-free evaluation gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from pufferlab.application.evaluation_gates import (
    GateApplicationStatus,
    evaluate_durable_gate,
)
from pufferlab.contracts.gates import (
    GateAggregateDeltaCheck,
    GateCandidateErrorRateCheck,
    GatePairedQueryCoverageCheck,
    GatePerQueryDropCheck,
    GatePolicy,
    GateReport,
    GateVerdict,
)

_INVALID_LINE = "error: evaluation gate policy or evidence is invalid"
_INTERNAL_LINE = "error: evaluation gate failed"


@dataclass(frozen=True, slots=True)
class GateCliExecution:
    """Bounded output already separated by stdout/stderr policy."""

    exit_code: int
    output_lines: tuple[str, ...] = ()
    error_lines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.exit_code not in {0, 1, 2, 4}:
            raise ValueError("gate CLI exit is outside the frozen domain")
        if self.output_lines and self.error_lines:
            raise ValueError("gate CLI output must use exactly one stream")
        if self.exit_code in {0, 4}:
            if not self.output_lines or self.error_lines:
                raise ValueError("gate reports require stdout only")
        else:
            expected_error = (_INTERNAL_LINE,) if self.exit_code == 1 else (_INVALID_LINE,)
            if self.output_lines or self.error_lines != expected_error:
                raise ValueError("gate failures require the exact fixed stderr line")
        lines = (*self.output_lines, *self.error_lines)
        if len(lines) > 15 or any(
            type(line) is not str or not line or "\n" in line or len(line) > 16_384
            for line in lines
        ):
            raise ValueError("gate CLI output is not one bounded line set")


def run_gate_cli(
    *,
    database_path: Path,
    run_id: UUID,
    candidate_config_id: UUID,
    policy: GatePolicy,
    output_format: str,
) -> GateCliExecution:
    """Run and render one gate without allowing evidence-bearing exceptions to escape."""

    try:
        if output_format not in {"text", "json"}:
            return GateCliExecution(exit_code=2, error_lines=(_INVALID_LINE,))
        result = evaluate_durable_gate(
            database_path=database_path,
            run_id=run_id,
            candidate_config_id=candidate_config_id,
            policy=policy,
        )
        if result.status is GateApplicationStatus.INVALID:
            return GateCliExecution(exit_code=2, error_lines=(_INVALID_LINE,))
        if result.status is GateApplicationStatus.INTERNAL or result.report is None:
            return GateCliExecution(exit_code=1, error_lines=(_INTERNAL_LINE,))
        report = result.report
        lines = (report.model_dump_json(),) if output_format == "json" else _render_text(report)
        return GateCliExecution(
            exit_code=0 if report.verdict is GateVerdict.PASSED else 4,
            output_lines=lines,
        )
    except BaseException as caught:
        caught.__traceback__ = None
        caught.__context__ = None
        caught.__cause__ = None
        return GateCliExecution(exit_code=1, error_lines=(_INTERNAL_LINE,))


def _render_text(report: GateReport) -> tuple[str, ...]:
    lines = [
        " ".join(
            (
                "gate",
                f"verdict={report.verdict.value}",
                f"run_id={report.run_id}",
                f"baseline_config_id={report.baseline_config_id}",
                f"candidate_config_id={report.candidate_config_id}",
                f"metric={report.metric.value}",
            )
        )
    ]
    for check in report.checks:
        passed = str(check.passed).lower()
        if isinstance(check, GateCandidateErrorRateCheck):
            lines.append(
                " ".join(
                    (
                        f"check={check.code.value}",
                        f"passed={passed}",
                        f"failed_candidate_queries={check.failed_candidate_queries}",
                        f"sample_count={check.sample_count}",
                        f"observed_error_rate={check.observed_error_rate}",
                        f"max_error_rate={check.max_error_rate}",
                    )
                )
            )
        elif isinstance(check, GatePairedQueryCoverageCheck):
            lines.append(
                " ".join(
                    (
                        f"check={check.code.value}",
                        f"passed={passed}",
                        f"paired_query_count={check.paired_query_count}",
                        f"excluded_query_count={check.excluded_query_count}",
                        f"min_paired_queries={check.min_paired_queries}",
                    )
                )
            )
        elif isinstance(check, GateAggregateDeltaCheck):
            lines.append(
                " ".join(
                    (
                        f"check={check.code.value}",
                        f"passed={passed}",
                        f"paired_query_count={check.paired_query_count}",
                        f"observed_mean_delta={check.observed_mean_delta}",
                        f"min_delta={check.min_delta}",
                    )
                )
            )
        elif isinstance(check, GatePerQueryDropCheck):
            lines.append(
                " ".join(
                    (
                        f"check={check.code.value}",
                        f"passed={passed}",
                        f"paired_query_count={check.paired_query_count}",
                        f"max_query_drop={check.max_query_drop}",
                        f"violating_query_count={check.violating_query_count}",
                    )
                )
            )
            lines.extend(
                " ".join(
                    (
                        "violation=per_query_drop",
                        f"query_id={violation.query_id}",
                        f"observed_delta={violation.observed_delta}",
                    )
                )
                for violation in check.violations
            )
    return tuple(lines)
