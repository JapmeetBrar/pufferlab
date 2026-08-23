"""Subprocess-only ASGI fixture whose lifespan shutdown never cooperates."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from typing import Any

import uvicorn
from pufferlab.cli.serve import BoundedUvicornServer


async def app(
    scope: dict[str, Any],
    receive: Callable[[], Awaitable[dict[str, Any]]],
    send: Callable[[dict[str, Any]], Awaitable[None]],
) -> None:
    if scope["type"] == "lifespan":
        assert (await receive())["type"] == "lifespan.startup"
        await send({"type": "lifespan.startup.complete"})
        assert (await receive())["type"] == "lifespan.shutdown"
        while True:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                continue
    if scope["type"] == "http":
        await receive()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"status":"ready"}'})


def main() -> int:
    port = int(sys.argv[1])
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        factory=False,
        workers=1,
        reload=False,
        proxy_headers=False,
        lifespan="on",
        access_log=False,
        log_level="critical",
        timeout_graceful_shutdown=5,
    )
    server = BoundedUvicornServer(config, _hard_deadline_seconds=0.25)
    server.run()
    return 0 if server.signal_termination_seen else 1


if __name__ == "__main__":
    raise SystemExit(main())
