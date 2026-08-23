from __future__ import annotations

import math
import traceback
from collections.abc import Callable, Sequence
from uuid import UUID

import pytest
from pufferlab.contracts.gates import (
    GATE_CHECK_ORDER,
    GateMetricName,
    GatePolicy,
    GateReport,
    GateVerdict,
)
from pufferlab.evals import (
    EvaluationWarning,
    EvaluationWarningCode,
    GateEvaluationError,
    GateEvaluationErrorCode,
    QueryMetrics,
    QueryOutcome,
    evaluate_gate,
)

_RUN_ID = UUID(int=1001)
_BASELINE_CONFIG_ID = UUID(int=1002)
_CANDIDATE_CONFIG_ID = UUID(int=1003)
_QUERY_IDS = tuple(UUID(int=value) for value in range(1, 51))


def _metrics(
    value: float = 0.5,
    *,
    ndcg: float | None = None,
    recall: float | None = None,
    mrr: float | None = None,
    warnings: tuple[EvaluationWarning, ...] = (),
) -> QueryMetrics:
    return QueryMetrics(
        ndcg_at_10=value if ndcg is None else ndcg,
        recall_at_50=value if recall is None else recall,
        mrr_at_10=value if mrr is None else mrr,
        warnings=warnings,
    )


def _success(query_id: UUID, metrics: QueryMetrics | None = None) -> QueryOutcome:
    return QueryOutcome.succeeded(query_id=query_id, metrics=metrics or _metrics())


def _failure(query_id: UUID, error_code: str = "typed_failure") -> QueryOutcome:
    return QueryOutcome.failed(query_id=query_id, error_code=error_code)


def _no_qrels(query_id: UUID, *, message: str = "no positive judgments") -> QueryOutcome:
    return _success(
        query_id,
        QueryMetrics(
            ndcg_at_10=None,
            recall_at_50=None,
            mrr_at_10=None,
            warnings=(
                EvaluationWarning(
                    code=EvaluationWarningCode.NO_POSITIVE_QRELS,
                    message=message,
                ),
            ),
        ),
    )


def _outcomes(
    query_ids: Sequence[UUID] = _QUERY_IDS,
    *,
    metrics: QueryMetrics | None = None,
) -> list[QueryOutcome]:
    return [_success(query_id, metrics) for query_id in query_ids]


def _report(
    *,
    expected_query_ids: Sequence[UUID] = _QUERY_IDS,
    baseline_outcomes: Sequence[QueryOutcome] | None = None,
    candidate_outcomes: Sequence[QueryOutcome] | None = None,
    policy: GatePolicy | None = None,
) -> GateReport:
    return evaluate_gate(
        run_id=_RUN_ID,
        baseline_config_id=_BASELINE_CONFIG_ID,
        candidate_config_id=_CANDIDATE_CONFIG_ID,
        expected_query_ids=expected_query_ids,
        baseline_outcomes=_outcomes() if baseline_outcomes is None else baseline_outcomes,
        candidate_outcomes=_outcomes() if candidate_outcomes is None else candidate_outcomes,
        policy=GatePolicy() if policy is None else policy,
    )


def _with_metric(metrics: QueryMetrics, metric: GateMetricName, value: float) -> QueryMetrics:
    if metric is GateMetricName.NDCG_AT_10:
        return _metrics(ndcg=value, recall=metrics.recall_at_50, mrr=metrics.mrr_at_10)
    if metric is GateMetricName.RECALL_AT_50:
        return _metrics(ndcg=metrics.ndcg_at_10, recall=value, mrr=metrics.mrr_at_10)
    return _metrics(ndcg=metrics.ndcg_at_10, recall=metrics.recall_at_50, mrr=value)


def test_all_four_threshold_equalities_pass_with_hand_calculated_evidence() -> None:
    baseline = _outcomes(metrics=_metrics(0.5))
    candidate = [
        _failure(query_id) if index < 10 else _success(query_id, _metrics(0.25))
        for index, query_id in enumerate(_QUERY_IDS)
    ]

    report = _report(
        baseline_outcomes=baseline,
        candidate_outcomes=candidate,
        policy=GatePolicy(
            min_delta=-0.25,
            max_query_drop=0.25,
            max_error_rate=0.2,
            min_paired_queries=40,
        ),
    )

    assert report.verdict is GateVerdict.PASSED
    assert tuple(check.code for check in report.checks) == GATE_CHECK_ORDER
    error_rate, coverage, aggregate, per_query = report.checks
    assert (error_rate.passed, error_rate.failed_candidate_queries) == (True, 10)
    assert error_rate.observed_error_rate == error_rate.max_error_rate == 0.2
    assert (coverage.passed, coverage.paired_query_count, coverage.excluded_query_count) == (
        True,
        40,
        10,
    )
    assert coverage.paired_query_count == coverage.min_paired_queries
    assert aggregate.observed_mean_delta == aggregate.min_delta == -0.25
    assert per_query.passed is True
    assert per_query.max_query_drop == 0.25
    assert per_query.violating_query_count == 0


def test_exact_paired_mean_does_not_hide_one_large_query_regression() -> None:
    baseline = [_success(_QUERY_IDS[0], _metrics(1.0))] + [
        _success(query_id, _metrics(0.0)) for query_id in _QUERY_IDS[1:]
    ]
    candidate = [_success(_QUERY_IDS[0], _metrics(0.0))] + [
        _success(query_id, _metrics(1.0)) for query_id in _QUERY_IDS[1:]
    ]

    report = _report(
        baseline_outcomes=baseline,
        candidate_outcomes=candidate,
        policy=GatePolicy(min_delta=0.0, max_query_drop=0.5),
    )

    assert report.verdict is GateVerdict.POLICY_FAILED
    assert report.checks[2].passed is True
    assert report.checks[2].observed_mean_delta == 0.96
    per_query = report.checks[3]
    assert per_query.passed is False
    assert per_query.violating_query_count == 1
    assert per_query.violations[0].query_id == _QUERY_IDS[0]
    assert per_query.violations[0].observed_delta == -1.0


def test_aggregate_uses_mean_of_pairs_not_difference_of_independent_means() -> None:
    baseline = [_success(_QUERY_IDS[0], _metrics(1.0))] + [
        _success(query_id, _metrics(0.0)) for query_id in _QUERY_IDS[1:]
    ]
    candidate = [_failure(_QUERY_IDS[0])] + [
        _success(query_id, _metrics(0.0)) for query_id in _QUERY_IDS[1:]
    ]

    report = _report(
        baseline_outcomes=baseline,
        candidate_outcomes=candidate,
        policy=GatePolicy(max_error_rate=0.02, min_paired_queries=49),
    )

    # Independent means would incorrectly report 0 - (1 / 50) = -0.02. The only 49 valid pairs
    # each have a zero delta, so their exact arithmetic mean is zero.
    assert report.verdict is GateVerdict.PASSED
    assert report.checks[2].paired_query_count == 49
    assert report.checks[2].observed_mean_delta == 0.0


def test_failures_and_no_qrels_have_exact_error_and_coverage_semantics() -> None:
    baseline = _outcomes()
    candidate = _outcomes()
    baseline[0] = _failure(_QUERY_IDS[0])
    candidate[1] = _failure(_QUERY_IDS[1])
    baseline[2] = _failure(_QUERY_IDS[2])
    candidate[2] = _failure(_QUERY_IDS[2])
    baseline[3] = _no_qrels(_QUERY_IDS[3])
    candidate[3] = _no_qrels(_QUERY_IDS[3])

    report = _report(
        baseline_outcomes=baseline,
        candidate_outcomes=candidate,
        policy=GatePolicy(max_error_rate=0.04, min_paired_queries=46),
    )

    assert report.verdict is GateVerdict.PASSED
    error_rate, coverage, aggregate, per_query = report.checks
    assert (error_rate.failed_candidate_queries, error_rate.observed_error_rate) == (2, 0.04)
    assert (coverage.paired_query_count, coverage.excluded_query_count) == (46, 4)
    assert aggregate.paired_query_count == per_query.paired_query_count == 46
    assert aggregate.observed_mean_delta == 0.0


def test_zero_pair_evidence_is_invalid_instead_of_a_policy_failure() -> None:
    successes = _outcomes()
    no_qrels = [_no_qrels(query_id) for query_id in _QUERY_IDS]
    failures = [_failure(query_id) for query_id in _QUERY_IDS]

    for baseline, candidate in (
        (no_qrels, no_qrels),
        (successes, failures),
        (failures, successes),
        (failures, failures),
    ):
        with pytest.raises(GateEvaluationError) as raised:
            _report(baseline_outcomes=baseline, candidate_outcomes=candidate)

        assert raised.value.code is GateEvaluationErrorCode.ZERO_PAIRED_QUERIES
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None


@pytest.mark.parametrize("metric", tuple(GateMetricName))
def test_selected_metric_alone_controls_aggregate_and_per_query_checks(
    metric: GateMetricName,
) -> None:
    base = _metrics(0.0, ndcg=0.0, recall=0.0, mrr=0.0)
    candidate = _metrics(1.0, ndcg=1.0, recall=1.0, mrr=1.0)
    base = _with_metric(base, metric, 0.75)
    candidate = _with_metric(candidate, metric, 0.25)

    report = _report(
        baseline_outcomes=_outcomes(metrics=base),
        candidate_outcomes=_outcomes(metrics=candidate),
        policy=GatePolicy(
            metric=metric,
            min_delta=-0.5,
            max_query_drop=0.5,
        ),
    )

    assert report.verdict is GateVerdict.PASSED
    assert report.checks[2].observed_mean_delta == -0.5
    assert report.checks[3].violating_query_count == 0


def test_property_every_exact_candidate_failure_rate_threshold_passes() -> None:
    baseline = _outcomes()
    for failed_count in range(50):
        candidate = [
            _failure(query_id) if index < failed_count else _success(query_id)
            for index, query_id in enumerate(_QUERY_IDS)
        ]
        exact_rate = failed_count / 50
        report = _report(
            baseline_outcomes=baseline,
            candidate_outcomes=candidate,
            policy=GatePolicy(
                max_error_rate=exact_rate,
                min_paired_queries=50 - failed_count,
            ),
        )

        assert report.verdict is GateVerdict.PASSED
        assert report.checks[0].failed_candidate_queries == failed_count
        assert report.checks[0].observed_error_rate == exact_rate
        assert report.checks[1].paired_query_count == 50 - failed_count

        if failed_count:
            below_exact_rate = math.nextafter(exact_rate, 0.0)
            failed = _report(
                baseline_outcomes=baseline,
                candidate_outcomes=candidate,
                policy=GatePolicy(
                    max_error_rate=below_exact_rate,
                    min_paired_queries=50 - failed_count,
                ),
            )
            assert failed.verdict is GateVerdict.POLICY_FAILED
            assert failed.checks[0].passed is False


@pytest.mark.parametrize("metric", tuple(GateMetricName))
def test_property_binary_fraction_metric_lattice_honors_inclusive_boundaries(
    metric: GateMetricName,
) -> None:
    values = (0.0, 0.25, 0.5, 0.75, 1.0)
    for baseline_value in values:
        for candidate_value in values:
            delta = candidate_value - baseline_value
            baseline_metrics = _with_metric(_metrics(), metric, baseline_value)
            candidate_metrics = _with_metric(_metrics(), metric, candidate_value)
            report = _report(
                baseline_outcomes=_outcomes(metrics=baseline_metrics),
                candidate_outcomes=_outcomes(metrics=candidate_metrics),
                policy=GatePolicy(
                    metric=metric,
                    min_delta=delta,
                    max_query_drop=max(0.0, -delta),
                ),
            )

            assert report.verdict is GateVerdict.PASSED
            assert report.checks[2].observed_mean_delta == delta
            assert report.checks[3].violating_query_count == 0


def test_query_violations_are_globally_counted_bounded_and_deterministically_sorted() -> None:
    baseline = _outcomes(metrics=_metrics(1.0))
    violating_values = (0.2, 0.1, 0.1, 0.4, 0.0, 0.3, 0.2, 0.0, 0.4, 0.3, 0.1, 0.2)
    candidate = [
        _success(query_id, _metrics(violating_values[index] if index < 12 else 1.0))
        for index, query_id in enumerate(_QUERY_IDS)
    ]

    report = _report(
        baseline_outcomes=list(reversed(baseline)),
        candidate_outcomes=list(reversed(candidate)),
        policy=GatePolicy(min_delta=-1.0, max_query_drop=0.5),
    )

    per_query = report.checks[3]
    expected = sorted(
        (
            (value - 1.0, query_id)
            for query_id, value in zip(_QUERY_IDS[:12], violating_values, strict=True)
        ),
        key=lambda item: (item[0], str(item[1])),
    )
    assert per_query.violating_query_count == 12
    assert len(per_query.violations) == 10
    assert [(row.observed_delta, row.query_id) for row in per_query.violations] == expected[:10]


def test_property_input_permutations_do_not_change_report_bytes() -> None:
    baseline = _outcomes()
    candidate = [
        _success(query_id, _metrics((index % 5) / 4)) for index, query_id in enumerate(_QUERY_IDS)
    ]
    candidate[0] = _failure(_QUERY_IDS[0])
    baseline[1] = _failure(_QUERY_IDS[1])
    candidate[2] = _no_qrels(_QUERY_IDS[2])
    baseline[2] = _no_qrels(_QUERY_IDS[2])
    policy = GatePolicy(
        min_delta=-1.0,
        max_query_drop=1.0,
        max_error_rate=1.0,
        min_paired_queries=1,
    )
    expected = _report(
        baseline_outcomes=baseline,
        candidate_outcomes=candidate,
        policy=policy,
    ).model_dump_json()

    for offset in range(50):
        rotated_baseline = baseline[offset:] + baseline[:offset]
        rotated_candidate = candidate[-offset:] + candidate[:-offset] if offset else candidate[:]
        actual = _report(
            expected_query_ids=_QUERY_IDS[offset:] + _QUERY_IDS[:offset],
            baseline_outcomes=rotated_baseline,
            candidate_outcomes=rotated_candidate,
            policy=policy,
        )
        assert actual.model_dump_json() == expected


def _replace_last_with_extra(outcomes: list[QueryOutcome]) -> list[QueryOutcome]:
    return [*outcomes[:-1], _success(UUID(int=999))]


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (lambda values: values[:-1], GateEvaluationErrorCode.INVALID_BASELINE_EVIDENCE),
        (
            lambda values: [*values, _success(UUID(int=999))],
            GateEvaluationErrorCode.INVALID_BASELINE_EVIDENCE,
        ),
        (
            lambda values: [*values[:-1], values[0]],
            GateEvaluationErrorCode.INVALID_BASELINE_EVIDENCE,
        ),
        (_replace_last_with_extra, GateEvaluationErrorCode.INVALID_BASELINE_EVIDENCE),
    ],
)
def test_baseline_missing_extra_duplicate_and_substituted_identities_fail_closed(
    mutate: Callable[[list[QueryOutcome]], list[QueryOutcome]],
    expected_code: GateEvaluationErrorCode,
) -> None:
    with pytest.raises(GateEvaluationError) as raised:
        _report(baseline_outcomes=mutate(_outcomes()))

    assert raised.value.code is expected_code


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values[:-1],
        lambda values: [*values, _success(UUID(int=999))],
        lambda values: [*values[:-1], values[0]],
        _replace_last_with_extra,
    ],
)
def test_candidate_missing_extra_duplicate_and_substituted_identities_fail_closed(
    mutate: Callable[[list[QueryOutcome]], list[QueryOutcome]],
) -> None:
    with pytest.raises(GateEvaluationError) as raised:
        _report(candidate_outcomes=mutate(_outcomes()))

    assert raised.value.code is GateEvaluationErrorCode.INVALID_CANDIDATE_EVIDENCE


@pytest.mark.parametrize(
    ("expected_query_ids", "expected_code"),
    [
        (_QUERY_IDS[:-1], GateEvaluationErrorCode.INVALID_QUERY_CATALOG),
        ((*_QUERY_IDS, UUID(int=999)), GateEvaluationErrorCode.INVALID_QUERY_CATALOG),
        ((*_QUERY_IDS[:-1], _QUERY_IDS[0]), GateEvaluationErrorCode.INVALID_QUERY_CATALOG),
        (
            (*_QUERY_IDS[:-1], UUID(int=999)),
            GateEvaluationErrorCode.INVALID_BASELINE_EVIDENCE,
        ),
    ],
)
def test_expected_catalog_count_duplicates_missing_and_extra_fail_closed(
    expected_query_ids: Sequence[UUID],
    expected_code: GateEvaluationErrorCode,
) -> None:
    with pytest.raises(GateEvaluationError) as raised:
        _report(expected_query_ids=expected_query_ids)

    assert raised.value.code is expected_code


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_delta", math.nan),
        ("min_delta", math.inf),
        ("max_query_drop", math.nan),
        ("max_query_drop", math.inf),
        ("max_error_rate", math.nan),
        ("max_error_rate", math.inf),
    ],
)
def test_forged_non_finite_policy_fails_before_report_construction(
    field: str,
    value: float,
) -> None:
    forged_policy = GatePolicy.model_construct(**{field: value})
    with pytest.raises(GateEvaluationError) as policy_error:
        _report(policy=forged_policy)
    assert policy_error.value.code is GateEvaluationErrorCode.INVALID_POLICY


def test_forged_non_finite_evidence_fails_before_report_construction() -> None:
    metrics = _metrics()
    object.__setattr__(metrics, "ndcg_at_10", math.inf)
    candidate = _outcomes()
    candidate[0] = _success(_QUERY_IDS[0], metrics)
    with pytest.raises(GateEvaluationError) as evidence_error:
        _report(candidate_outcomes=candidate)
    assert evidence_error.value.code is GateEvaluationErrorCode.INVALID_CANDIDATE_EVIDENCE

    outcome = candidate[1]
    object.__setattr__(outcome, "latency_ms", math.nan)
    candidate[0] = _success(_QUERY_IDS[0])
    with pytest.raises(GateEvaluationError) as latency_error:
        _report(candidate_outcomes=candidate)
    assert latency_error.value.code is GateEvaluationErrorCode.INVALID_CANDIDATE_EVIDENCE


def test_invalid_config_binding_fails_closed() -> None:
    with pytest.raises(GateEvaluationError) as same_config:
        evaluate_gate(
            run_id=_RUN_ID,
            baseline_config_id=_BASELINE_CONFIG_ID,
            candidate_config_id=_BASELINE_CONFIG_ID,
            expected_query_ids=_QUERY_IDS,
            baseline_outcomes=_outcomes(),
            candidate_outcomes=_outcomes(),
            policy=GatePolicy(),
        )
    assert same_config.value.code is GateEvaluationErrorCode.INVALID_BINDING


def test_report_and_safe_invalid_evidence_error_never_retain_free_form_evidence() -> None:
    marker = "licensed-query-text-and-provider-payload-marker"
    baseline = _outcomes()
    baseline[0] = _success(
        _QUERY_IDS[0],
        _metrics(
            warnings=(
                EvaluationWarning(
                    code=EvaluationWarningCode.DUPLICATE_QREL,
                    message=marker,
                ),
            )
        ),
    )
    candidate = _outcomes()
    candidate[0] = _failure(_QUERY_IDS[0], marker)

    report = _report(
        baseline_outcomes=baseline,
        candidate_outcomes=candidate,
        policy=GatePolicy(max_error_rate=1.0, min_paired_queries=49),
    )
    assert marker not in repr(report)
    assert marker not in report.model_dump_json()

    candidate[-1] = candidate[0]
    with pytest.raises(GateEvaluationError) as raised:
        _report(
            baseline_outcomes=baseline,
            candidate_outcomes=candidate,
            policy=GatePolicy(max_error_rate=1.0, min_paired_queries=49),
        )

    error = raised.value
    assert error.code is GateEvaluationErrorCode.INVALID_CANDIDATE_EVIDENCE
    assert error.__cause__ is None
    assert error.__context__ is None
    assert marker not in str(error)
    assert marker not in repr(error)
    assert marker not in "".join(traceback.format_exception(error))
    current = error.__traceback__
    while current is not None:
        if current.tb_frame.f_code.co_filename.endswith("/pufferlab/evals/gates.py"):
            assert marker not in repr(current.tb_frame.f_locals)
        current = current.tb_next
