import subprocess
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from pufferlab.config import Settings
from pufferlab.contracts.evals import EvalRunStatus
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.persistence.database import sqlite_url
from pufferlab.persistence.models import Base
from sqlalchemy import create_engine, inspect

from .helpers import SampleGraph, make_outcome, persist_graph

APPLICATION_TABLES = {
    "dataset_versions",
    "retrieval_configs",
    "query_sets",
    "judged_queries",
    "judged_document_titles",
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


def test_populated_0001_catalog_preserves_evidence_and_null_title_fallback(
    tmp_path: Path,
    sample_graph: SampleGraph,
) -> None:
    database_path = tmp_path / "populated-migration.sqlite3"
    config = Config("alembic.ini")
    config.set_main_option(
        "sqlalchemy.url", sqlite_url(database_path).render_as_string(hide_password=False)
    )
    command.upgrade(config, "20260822_0001")

    legacy_database = Database(database_path)
    try:
        legacy_repository = PufferLabRepository(legacy_database.session_factory)
        persist_graph(legacy_repository, sample_graph)
        run = sample_graph.make_run("populated-0001-run")
        legacy_repository.create_run(run)
        legacy_repository.transition_run(run.id, EvalRunStatus.RUNNING)
        outcome = make_outcome(
            run,
            sample_graph.configs[0].id,
            sample_graph.queries[0].id,
        )
        legacy_repository.record_outcome(outcome)
        expected_query_set = legacy_repository.get_query_set(sample_graph.query_set.id)
        expected_run = legacy_repository.get_run(run.id)
        expected_outcome = legacy_repository.get_outcome(
            run.id,
            outcome.config_id,
            outcome.query_id,
        )
    finally:
        legacy_database.dispose()

    command.upgrade(config, "20260825_0002")

    upgraded_database = Database(database_path)
    try:
        upgraded_repository = PufferLabRepository(upgraded_database.session_factory)
        assert upgraded_repository.get_query_set(sample_graph.query_set.id) == expected_query_set
        assert upgraded_repository.get_run(run.id) == expected_run
        assert (
            upgraded_repository.get_outcome(run.id, outcome.config_id, outcome.query_id)
            == expected_outcome
        )
        qrel_ids = [qrel.document_id for query in expected_query_set[1] for qrel in query.qrels]
        stored_titles = upgraded_repository.get_judged_document_titles(
            sample_graph.query_set.id,
            qrel_ids,
        )
        assert stored_titles == {}
        assert [stored_titles.get(document_id) for document_id in qrel_ids] == [None] * len(
            qrel_ids
        )
    finally:
        upgraded_database.dispose()
