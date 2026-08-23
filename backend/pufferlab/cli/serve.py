"""Bounded, loopback-only composition for the installed local API server."""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass
from types import FrameType
from typing import NoReturn, Protocol

import uvicorn

_ALLOWED_HOSTS = frozenset(("127.0.0.1", "::1", "localhost"))
_APP_IMPORT = "pufferlab.main:app"
_GRACEFUL_SHUTDOWN_SECONDS = 5
_PROCESS_SHUTDOWN_DEADLINE_SECONDS = 10.0
_SHUTDOWN_SIGNALS: tuple[signal.Signals, signal.Signals] = (signal.SIGINT, signal.SIGTERM)


@dataclass(frozen=True, slots=True)
class ServeOptions:
    """The complete caller-controlled surface for ``pufferlab serve``."""

    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        if self.host not in _ALLOWED_HOSTS:
            raise ValueError("serve host must be an exact allowlisted loopback name")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("serve port must be an integer from 1 through 65535")

    @property
    def bind_host(self) -> str:
        """Return the canonical address passed to Uvicorn."""

        if self.host == "localhost":
            return "127.0.0.1"
        return self.host


class _ServeServer(Protocol):
    @property
    def signal_termination_seen(self) -> bool: ...

    def run(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ServeDependencies:
    """Narrow construction seams used to verify the immutable server configuration."""

    config_factory: Callable[..., object]
    server_factory: Callable[[object], _ServeServer]


@dataclass(frozen=True, slots=True)
class ServeExecution:
    """A finite result that cannot carry an exception or runtime configuration value."""

    signal_termination_seen: bool
    synchronous_run_returned: bool

    @property
    def gracefully_stopped(self) -> bool:
        return self.signal_termination_seen and self.synchronous_run_returned

    @property
    def exit_code(self) -> int:
        return 0 if self.gracefully_stopped else 1


class BoundedUvicornServer(uvicorn.Server):
    """Bound the whole synchronous runner after an operator shutdown signal.

    Uvicorn bounds connection and task draining. Its lifespan shutdown and the outer
    ``asyncio.Runner`` teardown can still block indefinitely, so the first supported signal starts
    one daemon watchdog. The watchdog exits only if the outermost synchronous ``run`` has not
    returned before the fixed hard deadline.
    """

    def __init__(
        self,
        config: uvicorn.Config,
        *,
        _hard_exit: Callable[[int], NoReturn] = os._exit,
        _hard_deadline_seconds: float = _PROCESS_SHUTDOWN_DEADLINE_SECONDS,
    ) -> None:
        super().__init__(config)
        if _hard_deadline_seconds <= 0:
            raise ValueError("hard shutdown deadline must be positive")
        self._run_done = threading.Event()
        self._signal_termination_seen = False
        self._watchdog_started = False
        self._hard_exit = _hard_exit
        self._hard_deadline_seconds = _hard_deadline_seconds

    @property
    def signal_termination_seen(self) -> bool:
        return self._signal_termination_seen

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        """Request graceful stop once and force Uvicorn draining on a repeated signal."""

        del frame
        if sig not in _SHUTDOWN_SIGNALS:
            return
        if self._signal_termination_seen:
            self.force_exit = True
            return
        self._signal_termination_seen = True
        self.should_exit = True
        self._start_shutdown_watchdog()

    def run(self, sockets: list[socket.socket] | None = None) -> None:
        """Install handlers outside Uvicorn's asyncio runner and signal full teardown."""

        with self._outer_signal_handlers():
            try:
                super().run(sockets=sockets)
            finally:
                # ``super().run`` returns only after asyncio.Runner teardown. The watchdog must
                # remain armed until this exact boundary, not merely until Uvicorn closes sockets.
                self._run_done.set()

    def _start_shutdown_watchdog(self) -> None:
        if self._watchdog_started:
            return
        self._watchdog_started = True
        watchdog = threading.Thread(
            target=self._enforce_shutdown_deadline,
            name="pufferlab-serve-shutdown-watchdog",
            daemon=True,
        )
        try:
            watchdog.start()
        except BaseException:
            # There is no remaining mechanism capable of enforcing the advertised process bound.
            self._hard_exit(0)

    def _enforce_shutdown_deadline(self) -> None:
        if not self._run_done.wait(self._hard_deadline_seconds):
            self._hard_exit(0)

    @contextlib.contextmanager
    def _outer_signal_handlers(self) -> Generator[None, None, None]:
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        original_handlers: dict[
            signal.Signals,
            Callable[[int, FrameType | None], object] | int | None,
        ] = {}
        try:
            for handled_signal in _SHUTDOWN_SIGNALS:
                original_handlers[handled_signal] = signal.signal(
                    handled_signal,
                    self.handle_exit,
                )
            yield
        finally:
            for handled_signal, original_handler in original_handlers.items():
                signal.signal(handled_signal, original_handler)


def default_serve_dependencies() -> ServeDependencies:
    """Compose Uvicorn only after the serve command has parsed successfully."""

    def server_factory(config: object) -> BoundedUvicornServer:
        if not isinstance(config, uvicorn.Config):
            raise TypeError("serve configuration factory returned an invalid value")
        return BoundedUvicornServer(config)

    return ServeDependencies(
        config_factory=uvicorn.Config,
        server_factory=server_factory,
    )


def run_serve(
    options: ServeOptions,
    *,
    dependencies: ServeDependencies | None = None,
) -> ServeExecution:
    """Run the fixed server and collapse every completion path to one safe result bit."""

    server: _ServeServer | None = None
    synchronous_run_returned = False
    try:
        resolved = dependencies or default_serve_dependencies()
        config = resolved.config_factory(
            _APP_IMPORT,
            host=options.bind_host,
            port=options.port,
            factory=False,
            workers=1,
            reload=False,
            proxy_headers=False,
            lifespan="on",
            access_log=False,
            log_level="critical",
            timeout_graceful_shutdown=_GRACEFUL_SHUTDOWN_SECONDS,
        )
        server = resolved.server_factory(config)
        server.run()
        synchronous_run_returned = True
    except BaseException:
        pass

    signal_seen = False
    if server is not None:
        try:
            signal_seen = server.signal_termination_seen
        except BaseException:
            synchronous_run_returned = False
    return ServeExecution(
        signal_termination_seen=signal_seen,
        synchronous_run_returned=synchronous_run_returned,
    )


def render_serve_start(options: ServeOptions) -> str:
    """Render the sole variable line, restricted to already validated loopback coordinates."""

    return f"serve starting host={options.bind_host} port={options.port}"


def render_serve_finish(execution: ServeExecution) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return allowlisted stdout and stderr lines for the finite result."""

    if execution.gracefully_stopped:
        return (("serve stopped",), ())
    return ((), ("error: serve failed",))
