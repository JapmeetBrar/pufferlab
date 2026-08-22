"""Prove the local API key is absent from Git and production browser artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pufferlab.config import Settings

_ROOT = Path(__file__).parents[1]
_FRONTEND_SOURCE = _ROOT / "web" / "src"
_FRONTEND_BUILD = _ROOT / "web" / "dist"
_BANNED_BROWSER_NAMES = (
    b"authorization",
    b"query_vector",
    b"turbopuffer-api-key",
    b"turbopuffer_api_key",
)


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", *arguments),
        cwd=_ROOT,
        check=check,
        capture_output=True,
    )


def _tracked_paths() -> tuple[Path, ...]:
    result = _git("ls-files", "-z")
    return tuple(
        _ROOT / raw.decode()
        for raw in result.stdout.split(b"\0")
        if raw and (_ROOT / raw.decode()).is_file()
    )


def _production_browser_paths() -> tuple[Path, ...]:
    if not _FRONTEND_BUILD.is_dir():
        raise RuntimeError("web/dist is missing; build the production browser bundle first")
    source = tuple(
        path for path in _FRONTEND_SOURCE.rglob("*") if path.is_file() and ".test." not in path.name
    )
    build = tuple(path for path in _FRONTEND_BUILD.rglob("*") if path.is_file())
    return source + build


def verify() -> dict[str, int | str]:
    settings = Settings()
    secret = settings.turbopuffer_api_key
    if secret is None or not secret.get_secret_value():
        raise RuntimeError("TURBOPUFFER_API_KEY is required for exact secret scanning")
    key = secret.get_secret_value().encode()

    ignored = _git("check-ignore", "-q", ".env", check=False)
    tracked = _git("ls-files", "--error-unmatch", ".env", check=False)
    if ignored.returncode != 0 or tracked.returncode == 0:
        raise RuntimeError(".env must be ignored and untracked")

    tracked_paths = _tracked_paths()
    if any(key in path.read_bytes() for path in tracked_paths):
        raise RuntimeError("the local turbopuffer key appears in a tracked worktree file")

    object_ids = {
        line.split(maxsplit=1)[0]
        for line in _git("rev-list", "--objects", "--all").stdout.splitlines()
        if line
    }
    history_blobs = 0
    history_bytes = 0
    for object_id in object_ids:
        if _git("cat-file", "-t", object_id.decode()).stdout.strip() != b"blob":
            continue
        content = _git("cat-file", "blob", object_id.decode()).stdout
        history_blobs += 1
        history_bytes += len(content)
        if key in content:
            raise RuntimeError("the local turbopuffer key appears in Git object history")

    browser_paths = _production_browser_paths()
    for path in browser_paths:
        content = path.read_bytes().lower()
        if key.lower() in content:
            raise RuntimeError("the local turbopuffer key appears in a browser file")
        if any(name in content for name in _BANNED_BROWSER_NAMES):
            raise RuntimeError(f"a private credential/vector field appears in {path.name}")

    return {
        "secret_boundary_verification": "passed",
        "tracked_files_scanned": len(tracked_paths),
        "history_blobs_scanned": history_blobs,
        "history_bytes_scanned": history_bytes,
        "browser_files_scanned": len(browser_paths),
    }


def main() -> None:
    for name, value in verify().items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
