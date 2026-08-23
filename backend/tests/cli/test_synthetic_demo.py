from __future__ import annotations

import io
import socket
import sqlite3
from pathlib import Path
from typing import NoReturn

import pytest
from pufferlab.cli.main import main
from pufferlab.config import Settings


def _settings(data_dir: Path) -> Settings:
    return Settings.model_validate(
        {
            "pufferlab_data_dir": data_dir,
            "turbopuffer_api_key": None,
        }
    )


def test_demo_seed_help_describes_provider_free_idempotent_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["demo", "seed", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    normalized = " ".join(output.split())
    assert "requires no API key, model, provider, or network access" in normalized
    assert "Re-running it verifies the same durable state" in normalized


def test_direct_cli_seeds_clean_database_without_key_network_or_runtime_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "clean" / "data"
    output = io.StringIO()
    errors = io.StringIO()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network or provider-backed runtime construction is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    exit_code = main(
        ["demo", "seed"],
        settings_factory=lambda: _settings(data_dir),
        cli_application_factory=forbidden,
        stdout=output,
        stderr=errors,
    )

    assert exit_code == 0
    assert errors.getvalue() == ""
    rendered = output.getvalue()
    assert "data_origin=synthetic_demo" in rendered
    assert "timing_source=synthetic_unavailable" in rendered
    assert "queries=50 configs=4 outcomes=200 read_export_only=true" in rendered
    assert "secret" not in rendered.lower()
    assert (data_dir / "pufferlab.sqlite3").is_file()
    assert not list(data_dir.rglob("*.json"))
    with sqlite3.connect(data_dir / "pufferlab.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM query_outcomes").fetchone()[0] == 200


def test_direct_cli_rerun_keeps_one_run_and_stable_output(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    outputs: list[str] = []
    for _ in range(2):
        output = io.StringIO()
        assert main(["demo", "seed"], settings_factory=lambda: settings, stdout=output) == 0
        outputs.append(output.getvalue())

    assert outputs[0] == outputs[1]
    with sqlite3.connect(settings.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM query_outcomes").fetchone()[0] == 200


def test_cli_failure_is_redacted(tmp_path: Path) -> None:
    output = io.StringIO()
    errors = io.StringIO()

    def fail(_settings: Settings) -> NoReturn:
        raise RuntimeError("provider-secret-value")

    exit_code = main(
        ["demo", "seed"],
        settings_factory=lambda: _settings(tmp_path),
        synthetic_demo_seed_runner=fail,
        stdout=output,
        stderr=errors,
    )

    assert exit_code == 1
    assert output.getvalue() == ""
    assert errors.getvalue() == "error: synthetic demo seed failed\n"
    assert "provider-secret-value" not in errors.getvalue()
