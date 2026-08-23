"""Provider-free CLI composition for the deterministic offline demo."""

from __future__ import annotations

from collections.abc import Callable

from pufferlab.config import Settings
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.synthetic_demo import SyntheticDemoSeedResult, seed_synthetic_demo


def seed_synthetic_demo_database(settings: Settings) -> SyntheticDemoSeedResult:
    """Migrate and seed only the configured local SQLite database."""
    database = Database.from_settings(settings)
    try:
        database.migrate()
        return seed_synthetic_demo(PufferLabRepository(database.session_factory))
    finally:
        database.dispose()


def render_synthetic_demo(
    result: SyntheticDemoSeedResult,
    *,
    emit: Callable[[str], None],
) -> None:
    config_ids = ",".join(str(config.id) for config in result.configs)
    emit("synthetic demo ready data_origin=synthetic_demo timing_source=synthetic_unavailable")
    emit(f"dataset_id={result.dataset_version.id} query_set_id={result.query_set.id}")
    emit(f"run_id={result.run.id} config_ids={config_ids}")
    emit("queries=50 configs=4 outcomes=200 read_export_only=true")
