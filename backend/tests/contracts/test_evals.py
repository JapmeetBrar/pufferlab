from datetime import UTC, datetime
from uuid import uuid4

from pufferlab.contracts.evals import (
    EvalRun,
    EvalRunStatus,
    QuerySetSummary,
    RunEnvironment,
)


def test_eval_run_serializes_contract_version() -> None:
    eval_run = EvalRun(
        id=uuid4(),
        status=EvalRunStatus.QUEUED,
        query_set=QuerySetSummary(
            id=uuid4(),
            name="tiny queries",
            version="v1",
            query_count=1,
            content_hash="content-hash",
        ),
        baseline_config_id=uuid4(),
        candidate_config_ids=[uuid4()],
        summaries=[],
        completed_queries=0,
        total_queries=1,
        random_seed=20260822,
        environment=RunEnvironment(
            pufferlab_git_revision="test-revision",
            turbopuffer_region="gcp-us-central1",
            python_version="3.12",
            platform="test",
            max_concurrency=1,
            query_embedding_cache_enabled=False,
        ),
        created_at=datetime.now(UTC),
        started_at=None,
        completed_at=None,
        error=None,
    )

    assert eval_run.model_dump(mode="json")["contract_version"] == 1
