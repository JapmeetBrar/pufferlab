from __future__ import annotations

import fcntl
import io
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, cast
from urllib.error import URLError
from urllib.request import urlopen

import pytest
import uvicorn
from pufferlab.cli.main import main
from pufferlab.cli.serve import (
    BoundedUvicornServer,
    ServeDependencies,
    ServeOptions,
    run_serve,
)

ROOT = Path(__file__).resolve().parents[3]
HANG_FIXTURE = Path(__file__).with_name("serve_hang_fixture.py")
FORBIDDEN_PORTS = frozenset((8000, 5173))


class _SpyServer:
    def __init__(self, *, finish_from_signal: bool = True) -> None:
        self.signal_termination_seen = False
        self.finish_from_signal = finish_from_signal
        self.run_calls = 0

    def run(self) -> None:
        self.run_calls += 1
        self.signal_termination_seen = self.finish_from_signal


def _no_returning_callback(callback: Callable[[int], None]) -> Callable[[int], NoReturn]:
    return cast(Callable[[int], NoReturn], callback)


def _uvicorn_config() -> uvicorn.Config:
    async def empty_app(scope: object, receive: object, send: object) -> None:
        del scope, receive, send

    return uvicorn.Config(empty_app, log_config=None)


@pytest.mark.parametrize(
    "arguments",
    [
        ["serve", "--host", "0.0.0.0"],
        ["serve", "--host", "127.0.0.1 "],
        ["serve", "--host", "LOCALHOST"],
        ["serve", "--host", "localhost.local"],
        ["serve", "--host", "löcalhost"],
        ["serve", "--host", ""],
        ["serve", "--port", "0"],
        ["serve", "--port", "65536"],
        ["serve", "--port", "1.5"],
        ["serve", "--port", ""],
        ["serve", "--workers", "2"],
        ["serve", "--reload"],
    ],
)
def test_parser_rejects_invalid_serve_input_before_all_construction(
    arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_settings() -> NoReturn:
        pytest.fail("settings constructed before serve parser rejection")

    def forbidden_config(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail("server config constructed before serve parser rejection")

    def forbidden_server(config: object) -> _SpyServer:
        del config
        pytest.fail("server constructed before serve parser rejection")

    with pytest.raises(SystemExit) as error:
        main(
            arguments,
            settings_factory=forbidden_settings,
            cli_application_factory=lambda settings: pytest.fail(
                f"normal CLI application constructed: {settings!r}"
            ),
            serve_dependencies=ServeDependencies(forbidden_config, forbidden_server),
        )

    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("usage: pufferlab")


@pytest.mark.parametrize(
    ("submitted_host", "bound_host"),
    (("127.0.0.1", "127.0.0.1"), ("::1", "::1"), ("localhost", "127.0.0.1")),
)
def test_serve_uses_exact_immutable_uvicorn_configuration(
    submitted_host: str,
    bound_host: str,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sentinel_config = object()
    server = _SpyServer()

    def config_factory(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return sentinel_config

    def server_factory(config: object) -> _SpyServer:
        assert config is sentinel_config
        return server

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(
        ["serve", "--host", submitted_host, "--port", "43123"],
        settings_factory=lambda: pytest.fail("serve must not construct Settings in cli.main"),
        cli_application_factory=lambda settings: pytest.fail(
            f"serve constructed the normal CLI application: {settings!r}"
        ),
        serve_dependencies=ServeDependencies(config_factory, server_factory),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert calls == [
        (
            ("pufferlab.main:app",),
            {
                "host": bound_host,
                "port": 43123,
                "factory": False,
                "workers": 1,
                "reload": False,
                "proxy_headers": False,
                "lifespan": "on",
                "access_log": False,
                "log_level": "critical",
                "timeout_graceful_shutdown": 5,
            },
        )
    ]
    assert server.run_calls == 1
    assert stdout.getvalue() == f"serve starting host={bound_host} port=43123\nserve stopped\n"
    assert stderr.getvalue() == ""


def test_serve_collapses_every_base_exception_to_fixed_failure() -> None:
    class HostileFailure(BaseException):
        pass

    marker = "hostile-secret-path-marker"

    def fail_config(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise HostileFailure(marker)

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(
        ["serve", "--port", "43124"],
        serve_dependencies=ServeDependencies(fail_config, lambda config: _SpyServer()),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == "serve starting host=127.0.0.1 port=43124\n"
    assert stderr.getvalue() == "error: serve failed\n"
    assert marker not in stdout.getvalue() + stderr.getvalue()


@pytest.mark.parametrize("bad_port", [True, False, 0, 65536, 1.0])
def test_serve_options_reject_non_integer_or_out_of_range_port(bad_port: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        ServeOptions(port=bad_port)  # type: ignore[arg-type]


@pytest.mark.parametrize("received_signal", [signal.SIGINT, signal.SIGTERM])
def test_first_and_second_signal_use_one_daemon_watchdog(received_signal: signal.Signals) -> None:
    hard_exits: list[int] = []
    server = BoundedUvicornServer(
        _uvicorn_config(),
        _hard_exit=_no_returning_callback(hard_exits.append),
        _hard_deadline_seconds=1,
    )

    server.handle_exit(received_signal, None)
    watchdogs = [
        thread
        for thread in threading.enumerate()
        if thread.name == "pufferlab-serve-shutdown-watchdog"
    ]
    assert server.should_exit
    assert not server.force_exit
    assert server.signal_termination_seen
    assert server._captured_signals == []
    assert len(watchdogs) == 1
    assert watchdogs[0].daemon

    server.handle_exit(received_signal, None)
    assert server.force_exit
    assert server._captured_signals == []
    assert (
        len(
            [
                thread
                for thread in threading.enumerate()
                if thread.name == "pufferlab-serve-shutdown-watchdog"
            ]
        )
        == 1
    )
    server._run_done.set()
    watchdogs[0].join(timeout=1)
    assert hard_exits == []


def test_graceful_synchronous_run_disarms_watchdog_and_restores_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hard_exits: list[int] = []
    server = BoundedUvicornServer(
        _uvicorn_config(),
        _hard_exit=_no_returning_callback(hard_exits.append),
        _hard_deadline_seconds=0.05,
    )
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def graceful_run(instance: uvicorn.Server, sockets: list[socket.socket] | None = None) -> None:
        del sockets
        instance.handle_exit(signal.SIGTERM, None)

    monkeypatch.setattr(uvicorn.Server, "run", graceful_run)
    server.run()
    time.sleep(0.1)

    assert server.signal_termination_seen
    assert server._captured_signals == []
    assert hard_exits == []
    assert all(signal.getsignal(sig) is handler for sig, handler in original.items())


def test_hung_outer_run_triggers_exact_hard_exit() -> None:
    hard_exits: list[int] = []
    triggered = threading.Event()

    def hard_exit(code: int) -> None:
        hard_exits.append(code)
        triggered.set()

    server = BoundedUvicornServer(
        _uvicorn_config(),
        _hard_exit=_no_returning_callback(hard_exit),
        _hard_deadline_seconds=0.02,
    )
    server.handle_exit(signal.SIGINT, None)

    assert triggered.wait(1)
    assert hard_exits == [0]
    assert not server._run_done.is_set()


def test_watchdog_thread_start_failure_hard_exits_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hard_exits: list[int] = []
    server = BoundedUvicornServer(
        _uvicorn_config(),
        _hard_exit=_no_returning_callback(hard_exits.append),
    )

    def fail_start(thread: threading.Thread) -> None:
        del thread
        raise RuntimeError("hostile thread failure")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    server.handle_exit(signal.SIGTERM, None)

    assert hard_exits == [0]
    assert server.should_exit


def test_run_without_signal_starts_no_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    server = BoundedUvicornServer(_uvicorn_config())
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets=None: None)

    server.run()

    assert not server.signal_termination_seen
    assert not server._watchdog_started
    assert server._run_done.is_set()
    assert (
        run_serve(
            ServeOptions(port=43125),
            dependencies=ServeDependencies(
                lambda *args, **kwargs: object(),
                lambda config: _SpyServer(finish_from_signal=False),
            ),
        ).exit_code
        == 1
    )


def _allocated_port() -> int:
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = cast(int, probe.getsockname()[1])
        if port not in FORBIDDEN_PORTS:
            return port


def _wait_for_health(process: subprocess.Popen[str], port: int, *, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"server exited before health: stdout={stdout!r} stderr={stderr!r}")
        try:
            with urlopen(f"http://127.0.0.1:{port}/api/v1/health", timeout=0.2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.02)
    process.kill()
    stdout, stderr = process.communicate()
    pytest.fail(f"server did not become healthy: stdout={stdout!r} stderr={stderr!r}")


def _installed_command(
    data_dir: Path,
    port: int,
) -> subprocess.Popen[str]:
    executable = Path(sys.executable).with_name("pufferlab")
    assert executable.is_file()
    environment = os.environ.copy()
    environment["PUFFERLAB_DATA_DIR"] = str(data_dir)
    environment.pop("TURBOPUFFER_API_KEY", None)
    return subprocess.Popen(
        [str(executable), "serve", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _assert_socket_released(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


@pytest.mark.parametrize("shutdown_signal", [signal.SIGINT, signal.SIGTERM])
def test_installed_serve_health_signal_shutdown_and_resource_release(
    tmp_path: Path,
    shutdown_signal: signal.Signals,
) -> None:
    data_dir = tmp_path / "serve-data"
    port = _allocated_port()
    process = _installed_command(data_dir, port)
    try:
        _wait_for_health(process, port)
        started = time.monotonic()
        process.send_signal(shutdown_signal)
        stdout, stderr = process.communicate(timeout=11)
        elapsed = time.monotonic() - started
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    assert process.returncode == 0
    assert elapsed < 10.5
    assert stdout == f"serve starting host=127.0.0.1 port={port}\nserve stopped\n"
    assert stderr == ""
    _assert_socket_released(port)

    guard_path = data_dir / ".pufferlab-api.lock"
    guard_fd = os.open(guard_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        fcntl.flock(guard_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(guard_fd, fcntl.LOCK_UN)
    finally:
        os.close(guard_fd)


@pytest.mark.parametrize("shutdown_signal", [signal.SIGINT, signal.SIGTERM])
def test_hanging_lifespan_is_hard_bounded_without_output_leak(
    shutdown_signal: signal.Signals,
) -> None:
    port = _allocated_port()
    process = subprocess.Popen(
        [sys.executable, str(HANG_FIXTURE), str(port)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_health(process, port)
        started = time.monotonic()
        process.send_signal(shutdown_signal)
        stdout, stderr = process.communicate(timeout=2)
        elapsed = time.monotonic() - started
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    combined = stdout + stderr
    assert process.returncode == 0
    assert elapsed < 1.5
    assert "Traceback" not in combined
    assert str(HANG_FIXTURE) not in combined
    assert "serve_hang_fixture" not in combined
    _assert_socket_released(port)


def test_installed_serve_bind_failure_has_fixed_safe_output(tmp_path: Path) -> None:
    port = _allocated_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as owner:
        owner.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        owner.bind(("127.0.0.1", port))
        owner.listen()
        process = _installed_command(tmp_path / "bind-data", port)
        stdout, stderr = process.communicate(timeout=10)

    combined = stdout + stderr
    assert process.returncode == 1
    assert stdout == f"serve starting host=127.0.0.1 port={port}\n"
    assert stderr == "error: serve failed\n"
    assert "Traceback" not in combined
    assert str(tmp_path) not in combined


def test_installed_serve_startup_failure_has_fixed_safe_output(tmp_path: Path) -> None:
    port = _allocated_port()
    invalid_data_dir = tmp_path / "private-startup-marker"
    invalid_data_dir.write_text("not a directory", encoding="utf-8")
    process = _installed_command(invalid_data_dir, port)
    stdout, stderr = process.communicate(timeout=10)

    combined = stdout + stderr
    assert process.returncode == 1
    assert stdout == f"serve starting host=127.0.0.1 port={port}\n"
    assert stderr == "error: serve failed\n"
    assert "Traceback" not in combined
    assert str(invalid_data_dir) not in combined
    assert "private-startup-marker" not in combined
