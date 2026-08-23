"""SQLite engine/session lifecycle for PufferLab's local state."""

from pathlib import Path
from sqlite3 import Connection

from sqlalchemy import Engine, event
from sqlalchemy.engine import URL, create_engine
from sqlalchemy.orm import Session, sessionmaker

from pufferlab.config import Settings


def sqlite_url(path: Path) -> URL:
    return URL.create("sqlite+pysqlite", database=str(path.resolve()))


class Database:
    """Own one reusable engine and short-lived transaction-scoped sessions."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            sqlite_url(self.path),
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
        event.listen(self.engine, "connect", _configure_sqlite_connection)
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "Database":
        return cls(settings.database_path)

    def migrate(self) -> None:
        """Upgrade this database through the checked-in Alembic revision chain."""
        from pufferlab.persistence.migration import upgrade_database

        upgrade_database(self.path)

    def dispose(self) -> None:
        self.engine.dispose()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_: object) -> None:
        self.dispose()


def _configure_sqlite_connection(dbapi_connection: Connection, _: object) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


def dispose_engine(engine: Engine) -> None:
    """Small explicit hook for migration/test composition roots."""
    engine.dispose()
