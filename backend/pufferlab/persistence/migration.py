"""Programmatic Alembic entrypoint for local application startup."""

from pathlib import Path

from alembic import command
from alembic.config import Config

from pufferlab.persistence.database import sqlite_url


def upgrade_database(path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).parent / "migrations"),
    )
    config.set_main_option(
        "sqlalchemy.url",
        sqlite_url(path).render_as_string(hide_password=False),
    )
    command.upgrade(config, "head")
