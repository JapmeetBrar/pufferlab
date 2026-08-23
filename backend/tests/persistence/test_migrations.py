import subprocess
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from pufferlab.config import Settings
from pufferlab.persistence.database import sqlite_url
from pufferlab.persistence.models import Base
from sqlalchemy import create_engine, inspect

APPLICATION_TABLES = {
    "dataset_versions",
    "retrieval_configs",
    "query_sets",
    "judged_queries",
    "qrels",
    "eval_runs",
    "run_configs",
    "query_outcomes",
}


def test_database_path_is_fixed_under_configurable_data_dir(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, pufferlab_data_dir=tmp_path / "local-data")

    assert settings.database_path == tmp_path / "local-data" / "pufferlab.sqlite3"


def test_default_database_path_is_git_ignored() -> None:
    repository_root = Path(__file__).parents[3]

    subprocess.run(
        ["git", "check-ignore", "--quiet", "data/pufferlab.sqlite3"],
        cwd=repository_root,
        check=True,
    )


def test_initial_migration_upgrades_and_downgrades_clean_database(tmp_path: Path) -> None:
    database_path = tmp_path / "migration.sqlite3"
    config = Config("alembic.ini")
    config.set_main_option(
        "sqlalchemy.url", sqlite_url(database_path).render_as_string(hide_password=False)
    )

    command.upgrade(config, "head")
    engine = create_engine(sqlite_url(database_path))
    try:
        assert set(inspect(engine).get_table_names()) >= APPLICATION_TABLES
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            assert compare_metadata(context, Base.metadata) == []
    finally:
        engine.dispose()

    command.downgrade(config, "base")
    engine = create_engine(sqlite_url(database_path))
    try:
        assert APPLICATION_TABLES.isdisjoint(inspect(engine).get_table_names())
    finally:
        engine.dispose()
