"""Verify the public live comparison API without printing secrets or vectors."""

from __future__ import annotations

import argparse
import json
import math
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]

_QUERY_TEXT = "How can I find the program listening on port 8080?"
_FORBIDDEN_RESPONSE_KEYS = frozenset(
    {"api_key", "authorization", "query_vector", "turbopuffer_api_key", "vector"}
)


class VerificationError(RuntimeError):
    """A safe live-verification failure that excludes provider response bodies."""


def _as_object(value: object, *, label: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise VerificationError(f"{label} is not a JSON object")
    return cast(JsonObject, value)


def _as_list(value: JsonValue | None, *, label: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise VerificationError(f"{label} is not a JSON array")
    return value


def _required_string(value: JsonValue | None, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label} is not a non-empty string")
    return value


def _request_json(method: str, url: str, payload: JsonObject | None = None) -> JsonObject:
    body = None if payload is None else json.dumps(payload).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=120) as response:
            raw = response.read()
    except HTTPError as error:
        message = f"{method} {urlparse(url).path} returned HTTP {error.code}"
        raise VerificationError(message) from None
    except URLError:
        raise VerificationError(f"{method} {urlparse(url).path} could not connect") from None
    try:
        return _as_object(json.loads(raw), label=urlparse(url).path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise VerificationError(f"{method} {urlparse(url).path} returned invalid JSON") from None


def _validate_base_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise VerificationError("base URL must be an uncredentialed loopback HTTP origin")
    return value.rstrip("/")


def _reject_private_fields(value: JsonValue, *, path: str = "response") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in _FORBIDDEN_RESPONSE_KEYS:
                raise VerificationError(f"private field appeared at {path}.{key}")
            _reject_private_fields(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_private_fields(item, path=f"{path}[{index}]")


def _config_ids(config_payload: JsonObject) -> tuple[str, str]:
    if config_payload.get("contract_version") != 1:
        raise VerificationError("config response has the wrong contract version")
    configs = _as_list(config_payload.get("configs"), label="configs")
    by_mode: dict[str, str] = {}
    for item in configs:
        config = _as_object(item, label="config")
        mode = _required_string(config.get("mode"), label="config mode")
        identifier = _required_string(config.get("id"), label="config id")
        if mode in by_mode:
            raise VerificationError(f"config response contains duplicate {mode} modes")
        by_mode[mode] = identifier
    try:
        return by_mode["bm25"], by_mode["vector"]
    except KeyError:
        raise VerificationError("config response is missing BM25 or vector") from None


def _validate_result(
    result_value: JsonValue,
    *,
    expected_mode: str,
    expected_id: str,
) -> JsonObject:
    result = _as_object(result_value, label=f"{expected_mode} result")
    config = _as_object(result.get("config"), label=f"{expected_mode} config")
    if config.get("mode") != expected_mode or config.get("id") != expected_id:
        raise VerificationError(f"result identity does not preserve discovered {expected_mode}")
    hits = _as_list(result.get("hits"), label=f"{expected_mode} hits")
    if not hits:
        raise VerificationError(f"{expected_mode} returned no hits")
    for expected_rank, hit_value in enumerate(hits, start=1):
        hit = _as_object(hit_value, label=f"{expected_mode} hit")
        _required_string(hit.get("document_id"), label="document id")
        _required_string(hit.get("external_id"), label="external id")
        if hit.get("final_rank") != expected_rank:
            raise VerificationError(f"{expected_mode} hit ranks are not contiguous and 1-based")
        score = _as_object(hit.get("final_score"), label=f"{expected_mode} score")
        score_value = score.get("value")
        if (
            not isinstance(score_value, int | float)
            or isinstance(score_value, bool)
            or not math.isfinite(score_value)
        ):
            raise VerificationError(f"{expected_mode} score is not numeric")
        expected_kind = "bm25" if expected_mode == "bm25" else "vector_distance"
        expected_direction = "higher_is_better" if expected_mode == "bm25" else "lower_is_better"
        if (
            score.get("kind") != expected_kind
            or score.get("direction") != expected_direction
            or score.get("source") != "turbopuffer_dist"
        ):
            raise VerificationError(f"{expected_mode} score semantics are incorrect")
    timings = _as_list(result.get("timings"), label=f"{expected_mode} timings")
    timing_stages: set[str] = set()
    for item in timings:
        timing = _as_object(item, label="timing")
        timing_stages.add(_required_string(timing.get("stage"), label="timing stage"))
        duration = timing.get("duration_ms")
        if (
            not isinstance(duration, int | float)
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration < 0
            or timing.get("measurement") != "client_wall_clock"
        ):
            raise VerificationError(f"{expected_mode} timing semantics are incorrect")
    required_stages = {"turbopuffer"}
    if expected_mode == "vector":
        required_stages.add("embed")
    if not required_stages.issubset(timing_stages):
        raise VerificationError(f"{expected_mode} is missing a required timing stage")
    return result


def verify(base_url: str) -> JsonObject:
    base = _validate_base_url(base_url)
    health = _request_json("GET", f"{base}/api/v1/health")
    if health.get("contract_version") != 1 or health.get("status") != "ok":
        raise VerificationError("health response is not ready")

    configs = _request_json("GET", f"{base}/api/v1/configs")
    bm25_id, vector_id = _config_ids(configs)
    request_payload: JsonObject = {
        "contract_version": 1,
        "query_text": _QUERY_TEXT,
        "config_ids": [bm25_id, vector_id],
        "debug_provenance": True,
    }
    if set(request_payload) != {
        "contract_version",
        "query_text",
        "config_ids",
        "debug_provenance",
    }:
        raise VerificationError("compare request contains an unexpected field")
    response = _request_json("POST", f"{base}/api/v1/search/compare", request_payload)
    _reject_private_fields(response)
    if response.get("contract_version") != 1 or response.get("query_text") != _QUERY_TEXT:
        raise VerificationError("compare response identity is incorrect")
    results = _as_list(response.get("results"), label="results")
    if len(results) != 2:
        raise VerificationError("compare response does not contain exactly two results")
    bm25 = _validate_result(results[0], expected_mode="bm25", expected_id=bm25_id)
    vector = _validate_result(results[1], expected_mode="vector", expected_id=vector_id)

    def summary(result: JsonObject) -> JsonObject:
        hits = _as_list(result.get("hits"), label="summary hits")
        first = _as_object(hits[0], label="summary first hit")
        return {
            "config_id": _required_string(
                _as_object(result.get("config"), label="summary config").get("id"),
                label="summary config id",
            ),
            "hit_count": len(hits),
            "top_external_id": _required_string(first.get("external_id"), label="top external id"),
            "score": _as_object(first.get("final_score"), label="summary score"),
            "timings": _as_list(result.get("timings"), label="summary timings"),
        }

    return {
        "live_api_verification": "passed",
        "query_text": _QUERY_TEXT,
        "request_fields": ",".join(sorted(request_payload)),
        "bm25": summary(bm25),
        "vector": summary(vector),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    arguments = parser.parse_args()
    result = verify(arguments.base_url)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
