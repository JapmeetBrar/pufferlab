from __future__ import annotations

import io
import json
from pathlib import Path
from typing import NoReturn
from uuid import uuid4

import pufferlab.application.evaluation_gates as application_gate_module
import pufferlab.cli.gates as cli_gate_module
import pytest
from pufferlab.cli.gates import GateCliExecution, run_gate_cli
from pufferlab.cli.main import main
from pufferlab.config import Settings
from pufferlab.contracts.gates import GatePolicy, GateReport
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.synthetic_demo.seeder import SyntheticDemoSeedResult, seed_synthetic_demo


def _settings(data_dir: Path) -> Settings:
    return Settings.model_validate(
        {
            "pufferlab_data_dir": data_dir,
            "turbopuffer_api_key": "configured-secret-marker",
            "pufferlab_search_namespace": "namespace-secret-marker",
        }
    )


def _seed(settings: Settings) -> SyntheticDemoSeedResult:
    with Database(settings.database_path) as database:
        database.migrate()
        return seed_synthetic_demo(PufferLabRepository(database.session_factory))


def _arguments(
    seeded: SyntheticDemoSeedResult,
    *,
    candidate_index: int = 3,
    output_format: str = "text",
) -> list[str]:
    return [
        "eval",
        "gate",
        str(seeded.run.id),
        "--candidate",
        str(seeded.configs[candidate_index].id),
        "--min-delta",
        "-1",
        "--max-query-drop",
        "1",
        "--max-error-rate",
        "1",
        "--min-paired-queries",
        "49",
        "--format",
        output_format,
    ]


def test_gate_parser_builds_without_breaking_existing_export_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["eval", "gate", "--help"]) == 0
    with pytest.raises(SystemExit) as exit_info:
        main(["eval", "export", "--help"])
    assert exit_info.value.code == 0
    assert "--overwrite" in capsys.readouterr().out


def test_installed_gate_passes_provider_free_and_writes_stdout_once(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    seeded = _seed(settings)
    output = CountingStream()
    errors = CountingStream()

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("RuntimeCliApplication provider model search migration recovery")

    exit_code = main(
        _arguments(seeded),
        settings_factory=lambda: settings,
        cli_application_factory=forbidden,
        stdout=output,
        stderr=errors,
    )

    assert exit_code == 0
    assert output.write_calls == 1
    assert errors.write_calls == 0
    assert output.getvalue().startswith("gate verdict=passed ")
    assert "check=candidate_error_rate passed=true" in output.getvalue()
    assert "configured-secret-marker" not in output.getvalue()
    assert "namespace-secret-marker" not in output.getvalue()


def test_catalog_close_completes_before_render_and_single_stream_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path / "data")
    seeded = _seed(settings)
    events: list[str] = []
    real_open = application_gate_module.open_existing_read_only_catalog
    real_render = cli_gate_module._render_text

    class OrderedCatalog:
        def __init__(self, path: Path) -> None:
            self._inner = real_open(path)
            self.repository = self._inner.repository

        def close(self) -> None:
            self._inner.close()
            events.append("close")

    def render(report: GateReport) -> tuple[str, ...]:
        events.append("render")
        return real_render(report)

    monkeypatch.setattr(
        application_gate_module,
        "open_existing_read_only_catalog",
        OrderedCatalog,
    )
    monkeypatch.setattr(cli_gate_module, "_render_text", render)
    output = EventStream(events)
    exit_code = main(
        _arguments(seeded),
        settings_factory=lambda: settings,
        stdout=output,
        stderr=CountingStream(),
    )

    assert exit_code == 0
    assert events == ["close", "render", "write"]


def test_json_policy_failure_has_exact_exit_and_safe_contract(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    seeded = _seed(settings)
    output = CountingStream()
    errors = CountingStream()
    arguments = [
        "eval",
        "gate",
        str(seeded.run.id),
        "--candidate",
        str(seeded.configs[1].id),
        "--format",
        "json",
    ]

    exit_code = main(
        arguments,
        settings_factory=lambda: settings,
        stdout=output,
        stderr=errors,
    )

    assert exit_code == 4
    assert output.write_calls == 1
    assert errors.getvalue() == ""
    report = json.loads(output.getvalue())
    assert report["verdict"] == "policy_failed"
    assert [check["code"] for check in report["checks"]] == [
        "candidate_error_rate",
        "paired_query_coverage",
        "aggregate_delta",
        "per_query_drop",
    ]
    assert len(report["checks"][-1]["violations"]) <= 10
    assert "query_text" not in output.getvalue()
    assert "document_id" not in output.getvalue()
    assert "qrel" not in output.getvalue()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--min-delta", "2"),
        ("--min-delta", "nan"),
        ("--max-query-drop", "-0.1"),
        ("--max-error-rate", "inf"),
        ("--min-paired-queries", "51"),
    ],
)
def test_invalid_policy_is_exit_two_with_one_fixed_line(
    tmp_path: Path,
    option: str,
    value: str,
) -> None:
    settings = _settings(tmp_path / "data")
    seeded = _seed(settings)
    output = CountingStream()
    errors = CountingStream()
    arguments = _arguments(seeded)
    index = arguments.index(option)
    arguments[index + 1] = value

    exit_code = main(
        arguments,
        settings_factory=lambda: settings,
        stdout=output,
        stderr=errors,
    )

    assert exit_code == 2
    assert output.getvalue() == ""
    assert errors.getvalue() == "error: evaluation gate policy or evidence is invalid\n"
    assert errors.write_calls == 1


def test_direct_invalid_format_fails_closed_before_catalog_access(tmp_path: Path) -> None:
    execution = run_gate_cli(
        database_path=tmp_path / "missing-secret-path.sqlite3",
        run_id=uuid4(),
        candidate_config_id=uuid4(),
        policy=GatePolicy(),
        output_format="provider-secret-format",
    )
    assert execution == GateCliExecution(
        exit_code=2,
        error_lines=("error: evaluation gate policy or evidence is invalid",),
    )


@pytest.mark.parametrize("exit_code", [1, 2])
def test_execution_contract_rejects_marker_bearing_failure_lines(exit_code: int) -> None:
    with pytest.raises(ValueError, match="exact fixed stderr"):
        GateCliExecution(
            exit_code=exit_code,
            error_lines=("provider-path-secret-marker",),
        )


@pytest.mark.parametrize(
    "hostile_arguments",
    [
        ["provider-credential-marker"],
        ["--unknown-provider-marker"],
        ["--metric", "provider-secret-metric"],
        ["--min-delta", "provider-secret-number"],
    ],
)
def test_gate_parse_failures_discard_raw_argv_and_return_one_fixed_line(
    hostile_arguments: list[str],
) -> None:
    marker = "provider"
    output = CountingStream()
    errors = CountingStream()
    exit_code = main(
        ["eval", "gate", str(uuid4()), *hostile_arguments],
        stdout=output,
        stderr=errors,
    )

    assert exit_code == 2
    assert output.getvalue() == ""
    assert errors.getvalue() == "error: invalid evaluation gate arguments\n"
    assert marker not in errors.getvalue()
    assert errors.write_calls == 1


@pytest.mark.parametrize("control", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_runner_failure_is_fixed_internal_without_traceback_or_marker(
    tmp_path: Path,
    control: type[BaseException],
) -> None:
    settings = _settings(tmp_path / "data")
    seeded = _seed(settings)
    marker = "query-provider-secret-marker"
    caught = control(marker)

    def fail(**_kwargs: object) -> NoReturn:
        raise caught

    output = CountingStream()
    errors = CountingStream()
    exit_code = main(
        _arguments(seeded),
        settings_factory=lambda: settings,
        gate_runner=fail,
        stdout=output,
        stderr=errors,
    )

    assert exit_code == 1
    assert output.getvalue() == ""
    assert errors.getvalue() == "error: evaluation gate failed\n"
    assert marker not in errors.getvalue()
    assert caught.__traceback__ is None
    assert caught.__context__ is None
    assert caught.__cause__ is None


@pytest.mark.parametrize("output_format", ["text", "json"])
@pytest.mark.parametrize("control", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_render_failure_is_fixed_internal_and_detaches_sensitive_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_format: str,
    control: type[BaseException],
) -> None:
    settings = _settings(tmp_path / "data")
    seeded = _seed(settings)
    marker = "render-query-provider-secret-marker"
    caught = control(marker)

    if output_format == "text":

        def fail_text(_report: GateReport) -> NoReturn:
            raise caught

        monkeypatch.setattr(cli_gate_module, "_render_text", fail_text)
    else:

        def fail_json(self: GateReport, *_args: object, **_kwargs: object) -> NoReturn:
            del self
            raise caught

        monkeypatch.setattr(GateReport, "model_dump_json", fail_json)

    execution = run_gate_cli(
        database_path=settings.database_path,
        run_id=seeded.run.id,
        candidate_config_id=seeded.configs[3].id,
        policy=GatePolicy(
            min_delta=-1,
            max_query_drop=1,
            max_error_rate=1,
            min_paired_queries=49,
        ),
        output_format=output_format,
    )

    assert execution == GateCliExecution(
        exit_code=1,
        error_lines=("error: evaluation gate failed",),
    )
    assert marker not in repr(execution)
    assert caught.__traceback__ is None
    assert caught.__context__ is None
    assert caught.__cause__ is None


def test_output_and_error_stream_failures_never_escape_or_change_internal_exit(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "data")
    seeded = _seed(settings)
    errors = FailingStream()

    exit_code = main(
        _arguments(seeded),
        settings_factory=lambda: settings,
        stdout=FailingStream(),
        stderr=errors,
    )

    assert exit_code == 1
    assert errors.write_calls == 0


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
@pytest.mark.parametrize("after_write", [False, True])
@pytest.mark.parametrize("control", [BrokenPipeError, KeyboardInterrupt, SystemExit])
def test_each_selected_stream_is_written_at_most_once_across_process_control(
    tmp_path: Path,
    stream_name: str,
    after_write: bool,
    control: type[BaseException],
) -> None:
    settings = _settings(tmp_path / "data")
    seeded = _seed(settings)
    marker = "path-query-provider-secret-marker"
    caught = control(marker)
    hostile = RaisingStream(caught, after_write=after_write)
    safe = CountingStream()

    if stream_name == "stdout":
        output, errors = hostile, safe
        runner = None
    else:
        output, errors = safe, hostile

        def runner(**_kwargs: object) -> GateCliExecution:
            return GateCliExecution(
                exit_code=1,
                error_lines=("error: evaluation gate failed",),
            )

    exit_code = main(
        _arguments(seeded),
        settings_factory=lambda: settings,
        gate_runner=runner,
        stdout=output,
        stderr=errors,
    )

    assert exit_code == 1
    assert hostile.write_calls == 1
    assert safe.write_calls == 0
    assert caught.__traceback__ is None
    assert caught.__context__ is None
    assert caught.__cause__ is None
    assert str(settings.database_path) not in repr(caught)


class CountingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.write_calls = 0

    def write(self, value: str) -> int:
        self.write_calls += 1
        return super().write(value)


class FailingStream(CountingStream):
    def write(self, value: str) -> NoReturn:
        del value
        self.write_calls += 1
        raise OSError("stream-provider-secret-marker")


class EventStream(CountingStream):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def write(self, value: str) -> int:
        self._events.append("write")
        return super().write(value)


class RaisingStream(CountingStream):
    def __init__(self, error: BaseException, *, after_write: bool) -> None:
        super().__init__()
        self._error = error
        self._after_write = after_write

    def write(self, value: str) -> int:
        self.write_calls += 1
        if self._after_write:
            io.StringIO.write(self, value)
        raise self._error
