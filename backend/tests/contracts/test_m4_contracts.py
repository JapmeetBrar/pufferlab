from math import inf, nan
from typing import Any
from uuid import UUID

import pytest
from pufferlab.contracts.capabilities import (
    CAPABILITY_ACTION_ORDER,
    CAPABILITY_REQUIREMENT_ORDER,
    CapabilitiesResponse,
    CapabilityActionCode,
    CapabilityRequirementCode,
    CapabilityState,
    LivePlaygroundCapability,
)
from pufferlab.contracts.errors import ApiErrorCode
from pufferlab.contracts.gates import (
    GATE_CHECK_ORDER,
    GateAggregateDeltaCheck,
    GateCandidateErrorRateCheck,
    GateCheckCode,
    GateMetricName,
    GatePairedQueryCoverageCheck,
    GatePerQueryDropCheck,
    GatePolicy,
    GateQueryViolation,
    GateReport,
    GateVerdict,
)
from pufferlab.main import create_app
from pydantic import ValidationError


def _configured_capability() -> LivePlaygroundCapability:
    return LivePlaygroundCapability(
        state=CapabilityState.LOCALLY_CONFIGURED,
        requirements=(),
        next_action=None,
    )


def _passing_checks() -> tuple[
    GateCandidateErrorRateCheck,
    GatePairedQueryCoverageCheck,
    GateAggregateDeltaCheck,
    GatePerQueryDropCheck,
]:
    return (
        GateCandidateErrorRateCheck(
            passed=True,
            failed_candidate_queries=0,
            observed_error_rate=0.0,
            max_error_rate=0.0,
        ),
        GatePairedQueryCoverageCheck(
            passed=True,
            paired_query_count=50,
            excluded_query_count=0,
            min_paired_queries=50,
        ),
        GateAggregateDeltaCheck(
            passed=True,
            metric=GateMetricName.NDCG_AT_10,
            paired_query_count=50,
            observed_mean_delta=0.0,
            min_delta=0.0,
        ),
        GatePerQueryDropCheck(
            passed=True,
            metric=GateMetricName.NDCG_AT_10,
            paired_query_count=50,
            max_query_drop=0.2,
            violating_query_count=0,
            violations=(),
        ),
    )


def _passing_report() -> GateReport:
    return GateReport(
        verdict=GateVerdict.PASSED,
        run_id=UUID(int=1),
        baseline_config_id=UUID(int=2),
        candidate_config_id=UUID(int=3),
        metric=GateMetricName.NDCG_AT_10,
        checks=_passing_checks(),
    )


def _property_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            names.update(value)
        if isinstance(value, dict):
            names.update(_property_names(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    names.update(_property_names(item))
    return names


def _free_form_strings(schema: object) -> list[dict[str, Any]]:
    if isinstance(schema, list):
        return [entry for item in schema for entry in _free_form_strings(item)]
    if not isinstance(schema, dict):
        return []
    found = (
        [schema]
        if schema.get("type") == "string"
        and "enum" not in schema
        and "const" not in schema
        and schema.get("format") != "uuid"
        else []
    )
    return found + [entry for value in schema.values() for entry in _free_form_strings(value)]


def test_capability_codes_and_actions_have_frozen_allowlisted_order() -> None:
    assert CAPABILITY_REQUIREMENT_ORDER == (
        CapabilityRequirementCode.API_KEY,
        CapabilityRequirementCode.SEARCH_NAMESPACE,
        CapabilityRequirementCode.REGION,
        CapabilityRequirementCode.LIVE_SEARCH_RUNTIME,
        CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,
        CapabilityRequirementCode.OWNED_TINY_CREDENTIAL_MISMATCH,
        CapabilityRequirementCode.OWNED_TINY_REGION_MISMATCH,
    )
    assert CAPABILITY_ACTION_ORDER == (
        CapabilityActionCode.CONFIGURE_API_KEY,
        CapabilityActionCode.CONFIGURE_SEARCH_NAMESPACE,
        CapabilityActionCode.CONFIGURE_REGION,
        CapabilityActionCode.INSTALL_LIVE_SEARCH_RUNTIME,
        CapabilityActionCode.RESOLVE_OWNED_TINY_RECEIPT,
        CapabilityActionCode.USE_OWNED_TINY_CREDENTIAL,
        CapabilityActionCode.USE_OWNED_TINY_REGION,
    )


@pytest.mark.parametrize(
    ("requirement", "action"),
    list(zip(CAPABILITY_REQUIREMENT_ORDER, CAPABILITY_ACTION_ORDER, strict=True)),
)
def test_first_unmet_capability_requirement_selects_exact_action(
    requirement: CapabilityRequirementCode,
    action: CapabilityActionCode,
) -> None:
    capability = LivePlaygroundCapability(
        state=CapabilityState.ACTION_REQUIRED,
        requirements=(requirement,),
        next_action=action,
    )

    assert capability.model_dump(mode="json") == {
        "state": "action_required",
        "requirements": [requirement.value],
        "next_action": action.value,
    }


def test_capability_requirements_are_a_unique_ordered_subsequence() -> None:
    valid = LivePlaygroundCapability(
        state=CapabilityState.ACTION_REQUIRED,
        requirements=(
            CapabilityRequirementCode.API_KEY,
            CapabilityRequirementCode.REGION,
            CapabilityRequirementCode.OWNED_TINY_RECEIPT_INVALID,
        ),
        next_action=CapabilityActionCode.CONFIGURE_API_KEY,
    )
    assert valid.requirements[1] is CapabilityRequirementCode.REGION

    payload = valid.model_dump(mode="json")
    with pytest.raises(ValidationError, match="frozen contract order"):
        LivePlaygroundCapability.model_validate(
            {**payload, "requirements": list(reversed(payload["requirements"]))}
        )
    with pytest.raises(ValidationError, match="must be unique"):
        LivePlaygroundCapability.model_validate({**payload, "requirements": ["api_key", "api_key"]})


def test_capability_state_action_and_extra_fields_fail_closed() -> None:
    configured = _configured_capability()
    response = CapabilitiesResponse(live_playground=configured)

    assert response.model_dump(mode="json") == {
        "contract_version": 1,
        "live_playground": {
            "state": "locally_configured",
            "requirements": [],
            "next_action": None,
        },
    }
    with pytest.raises(ValidationError, match="requires no action"):
        LivePlaygroundCapability(
            state=CapabilityState.LOCALLY_CONFIGURED,
            requirements=(),
            next_action=CapabilityActionCode.CONFIGURE_API_KEY,
        )
    with pytest.raises(ValidationError, match="require action"):
        LivePlaygroundCapability(
            state=CapabilityState.LOCALLY_CONFIGURED,
            requirements=(CapabilityRequirementCode.API_KEY,),
            next_action=CapabilityActionCode.CONFIGURE_API_KEY,
        )
    with pytest.raises(ValidationError, match="first unmet"):
        LivePlaygroundCapability(
            state=CapabilityState.ACTION_REQUIRED,
            requirements=(CapabilityRequirementCode.API_KEY,),
            next_action=CapabilityActionCode.CONFIGURE_REGION,
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        CapabilitiesResponse.model_validate(
            {
                **response.model_dump(mode="json"),
                "namespace": "must-not-cross-contract",
            }
        )
    with pytest.raises(ValidationError, match="frozen"):
        configured.state = CapabilityState.ACTION_REQUIRED


def test_gate_policy_has_exact_finite_strict_domains() -> None:
    assert tuple(GateMetricName) == (
        GateMetricName.NDCG_AT_10,
        GateMetricName.RECALL_AT_50,
        GateMetricName.MRR_AT_10,
    )
    assert GatePolicy().model_dump(mode="json") == {
        "contract_version": 1,
        "metric": "ndcg@10",
        "min_delta": 0.0,
        "max_query_drop": 0.2,
        "max_error_rate": 0.0,
        "min_paired_queries": 50,
    }
    for field, value in (
        ("min_delta", -1.01),
        ("min_delta", 1.01),
        ("max_query_drop", -0.01),
        ("max_query_drop", 1.01),
        ("max_error_rate", -0.01),
        ("max_error_rate", 1.01),
        ("min_paired_queries", 0),
        ("min_paired_queries", 51),
        ("min_delta", nan),
        ("max_query_drop", inf),
        ("max_error_rate", -inf),
        ("min_delta", True),
        ("max_query_drop", "0.2"),
        ("min_paired_queries", True),
        ("min_paired_queries", "50"),
    ):
        with pytest.raises(ValidationError):
            GatePolicy.model_validate({field: value})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GatePolicy.model_validate({"namespace": "forbidden"})


def test_gate_report_freezes_check_order_and_inclusive_pass_boundaries() -> None:
    report = _passing_report()

    assert GATE_CHECK_ORDER == (
        GateCheckCode.CANDIDATE_ERROR_RATE,
        GateCheckCode.PAIRED_QUERY_COVERAGE,
        GateCheckCode.AGGREGATE_DELTA,
        GateCheckCode.PER_QUERY_DROP,
    )
    assert tuple(check.code for check in report.checks) == GATE_CHECK_ORDER
    assert report.verdict is GateVerdict.PASSED
    payload = report.model_dump(mode="json")
    payload["checks"][0], payload["checks"][1] = payload["checks"][1], payload["checks"][0]
    with pytest.raises(ValidationError):
        GateReport.model_validate(payload)


def test_gate_report_rejects_inconsistent_math_population_metric_and_verdict() -> None:
    report = _passing_report()
    payload = report.model_dump(mode="json")

    attacks = (
        {"checks.0.observed_error_rate": 0.02},
        {"checks.1.excluded_query_count": 1},
        {"checks.2.paired_query_count": 49},
        {"checks.2.metric": "mrr@10"},
        {"candidate_config_id": payload["baseline_config_id"]},
        {"verdict": "policy_failed"},
    )
    for attack in attacks:
        attacked = report.model_dump(mode="json")
        key, value = next(iter(attack.items()))
        if key.startswith("checks."):
            _, index, field = key.split(".")
            attacked["checks"][int(index)][field] = value
        else:
            attacked[key] = value
        with pytest.raises(ValidationError):
            GateReport.model_validate(attacked)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            GateCandidateErrorRateCheck,
            {
                "passed": True,
                "failed_candidate_queries": False,
                "observed_error_rate": 0.0,
                "max_error_rate": 0.0,
            },
        ),
        (
            GateCandidateErrorRateCheck,
            {
                "passed": True,
                "failed_candidate_queries": 0,
                "sample_count": "50",
                "observed_error_rate": 0.0,
                "max_error_rate": 0.0,
            },
        ),
        (
            GateCandidateErrorRateCheck,
            {
                "passed": True,
                "failed_candidate_queries": 0,
                "observed_error_rate": nan,
                "max_error_rate": 0.0,
            },
        ),
        (
            GatePairedQueryCoverageCheck,
            {
                "passed": True,
                "paired_query_count": "50",
                "excluded_query_count": 0,
                "min_paired_queries": 50,
            },
        ),
        (
            GateAggregateDeltaCheck,
            {
                "passed": True,
                "metric": "ndcg@10",
                "paired_query_count": 50,
                "observed_mean_delta": True,
                "min_delta": 0.0,
            },
        ),
        (
            GatePerQueryDropCheck,
            {
                "passed": True,
                "metric": "ndcg@10",
                "paired_query_count": 50,
                "max_query_drop": "0.2",
                "violating_query_count": 0,
                "violations": [],
            },
        ),
        (
            GateQueryViolation,
            {"query_id": str(UUID(int=1)), "observed_delta": inf},
        ),
    ],
)
def test_gate_report_numeric_fields_reject_bool_string_and_non_finite_input(
    model: Any,
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_policy_failure_report_bounds_and_orders_unique_query_violations() -> None:
    violations = tuple(
        GateQueryViolation(query_id=UUID(int=index), observed_delta=-0.5 + index / 1000)
        for index in range(1, 11)
    )
    checks = (
        GateCandidateErrorRateCheck(
            passed=False,
            failed_candidate_queries=1,
            observed_error_rate=0.02,
            max_error_rate=0.0,
        ),
        GatePairedQueryCoverageCheck(
            passed=False,
            paired_query_count=49,
            excluded_query_count=1,
            min_paired_queries=50,
        ),
        GateAggregateDeltaCheck(
            passed=False,
            metric=GateMetricName.RECALL_AT_50,
            paired_query_count=49,
            observed_mean_delta=-0.1,
            min_delta=0.0,
        ),
        GatePerQueryDropCheck(
            passed=False,
            metric=GateMetricName.RECALL_AT_50,
            paired_query_count=49,
            max_query_drop=0.2,
            violating_query_count=11,
            violations=violations,
        ),
    )
    report = GateReport(
        verdict=GateVerdict.POLICY_FAILED,
        run_id=UUID(int=100),
        baseline_config_id=UUID(int=101),
        candidate_config_id=UUID(int=102),
        metric=GateMetricName.RECALL_AT_50,
        checks=checks,
    )

    assert len(report.checks[-1].violations) == 10
    with pytest.raises(ValidationError, match="delta then UUID"):
        GatePerQueryDropCheck.model_validate(
            {**checks[-1].model_dump(mode="json"), "violations": list(reversed(violations))}
        )
    with pytest.raises(ValidationError, match="first ten"):
        GatePerQueryDropCheck.model_validate(
            {**checks[-1].model_dump(mode="json"), "violations": list(violations[:-1])}
        )
    with pytest.raises(ValidationError, match="threshold must pass"):
        GatePerQueryDropCheck.model_validate(
            {
                **checks[-1].model_dump(mode="json"),
                "violating_query_count": 1,
                "violations": [
                    {"query_id": str(UUID(int=1)), "observed_delta": -0.2},
                ],
            }
        )
    with pytest.raises(ValidationError, match="must be unique"):
        GatePerQueryDropCheck.model_validate(
            {
                **checks[-1].model_dump(mode="json"),
                "violating_query_count": 2,
                "violations": [violations[0].model_dump(), violations[0].model_dump()],
            }
        )


def test_m4_contract_schemas_have_no_sensitive_or_free_form_value_fields() -> None:
    schemas = (
        CapabilitiesResponse.model_json_schema(),
        GatePolicy.model_json_schema(),
        GateReport.model_json_schema(),
    )
    forbidden_fields = {
        "api_key",
        "credential",
        "details",
        "document_id",
        "message",
        "namespace",
        "path",
        "provider_response",
        "qrels",
        "query_text",
        "region",
        "secret",
        "traceback",
    }

    for schema in schemas:
        assert not (_property_names(schema) & forbidden_fields)
        assert _free_form_strings(schema) == []

    report = _passing_report()
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GateReport.model_validate({**report.model_dump(mode="json"), "path": "/private"})
    with pytest.raises(ValidationError, match="frozen"):
        report.metric = GateMetricName.MRR_AT_10


def test_only_reachable_error_enum_changes_openapi_at_contract_freeze() -> None:
    schema = create_app().openapi()
    components = schema["components"]["schemas"]

    assert "/api/v1/capabilities" not in schema["paths"]
    assert components["ApiErrorCode"]["enum"] == [code.value for code in ApiErrorCode]
    assert "configuration_required" in components["ApiErrorCode"]["enum"]
    assert not any(
        name.startswith(("Capability", "Capabilities", "LivePlayground", "Gate"))
        for name in components
    )
