from collections.abc import Iterator
from pathlib import Path

import pytest
from pufferlab.persistence import Database, PufferLabRepository

from .helpers import SampleGraph, make_sample_graph


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    database = Database(tmp_path / "state.sqlite3")
    database.migrate()
    try:
        yield database
    finally:
        database.dispose()


@pytest.fixture
def repository(database: Database) -> PufferLabRepository:
    return PufferLabRepository(database.session_factory)


@pytest.fixture
def sample_graph() -> SampleGraph:
    return make_sample_graph()
