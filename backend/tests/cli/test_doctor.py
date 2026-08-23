from __future__ import annotations

import hashlib
import io
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NoReturn
from uuid import UUID, uuid4, uuid5

import pytest
from pufferlab.cli.doctor import (
    DoctorCheck,
    DoctorCheckName,
    DoctorCheckState,
    DoctorDependencies,
    DoctorLiveTarget,
    DoctorMode,
    DoctorReport,
    default_doctor_dependencies,
    doctor_exit_code,
)
from pufferlab.cli.main import main
from pufferlab.config import Settings
from pufferlab.contracts.capabilities import (
    CapabilitiesResponse,
    CapabilityState,
    LivePlaygroundCapability,
)
from pufferlab.contracts.datasets import (
    DatasetStatus,
    DatasetVersion,
    FtsProfile,
    IndexProfile,
)
from pufferlab.contracts.evals import (
    EvalRun,
    EvalRunStatus,
    EvalSuccessPayload,
    JudgedQuery,
    PerQueryMetrics,
    Qrel,
    QuerySet,
    QuerySetSummary,
    RunEnvironment,
)
from pufferlab.datasets import load_unix_dataset_manifest
from pufferlab.datasets.cqadupstack import load_curated_query_manifest
from pufferlab.datasets.identity import PUFFERLAB_NAMESPACE_UUID
from pufferlab.datasets.schema import compile_namespace_write_spec
from pufferlab.datasets.unix_application import UNIX_REVISION_CREATED_AT
from pufferlab.jobs.eval_runner import encode_outcome_payload, finalize_durable_outcomes
from pufferlab.persistence import Database, PufferLabRepository
from pufferlab.persistence.types import QueryOutcome, QueryOutcomeStatus
from pufferlab.providers.metadata_probe import MetadataProbeResult, MetadataProbeState
from pufferlab.retrieval.config import derive_bound_retrieval_configs
from pufferlab.synthetic_demo.seeder import materialize_synthetic_demo
from pydantic import SecretStr

ROOT = Path(__file__).resolve().parents[3]
UNIX_MANIFEST = ROOT / "datasets" / "cqadupstack-unix" / "dataset-manifest.json"
CURATED_MANIFEST = ROOT / "datasets" / "cqadupstack-unix" / "curated-50.json"


def _settings(
    data_dir: Path,
    *,
    api_key: str | None = None,
    region: str = "gcp-us-west1",
) -> Settings:
    return Settings.model_validate(
        {
            "pufferlab_data_dir": data_dir,
            "turbopuffer_api_key": api_key,
            "turbopuffer_region": region,
        }
    )


def _seed(settings: Settings) -> None:
    assert main(["demo", "seed"], settings_factory=lambda: settings, stdout=io.StringIO()) == 0


def _snapshot(path: Path) -> tuple[str, int, int, int, tuple[str, ...]]:
    metadata = path.stat()
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        tuple(sorted(item.name for item in path.parent.iterdir())),
    )


def _configured_inspector(_settings: Settings) -> ConfiguredInspector:
    return ConfiguredInspector()


class ConfiguredInspector:
    def inspect(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            live_playground=LivePlaygroundCapability(
                state=CapabilityState.LOCALLY_CONFIGURED,
                requirements=(),
                next_action=None,
            )
        )


class FakeProbe:
    def __init__(
        self,
        state: MetadataProbeState | None = MetadataProbeState.INDEX_UP_TO_DATE,
        *,
        failure: BaseException | None = None,
    ) -> None:
        self.state = state
        self.failure = failure
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(
        self,
        *,
        api_key: str,
        region: str,
        namespace: str,
    ) -> MetadataProbeResult:
        self.calls.append((api_key, region, namespace))
        if self.failure is not None:
            raise self.failure
        assert self.state is not None
        return MetadataProbeResult(state=self.state)


def _live_tiny_dependencies(
    probe: FakeProbe,
    *,
    namespace: str = "pufferlab-live-marker",
) -> DoctorDependencies:
    return replace(
        default_doctor_dependencies(),
        capability_inspector_factory=_configured_inspector,
        owned_tiny_target_resolver=lambda _settings: DoctorLiveTarget(
            namespace=namespace,
            region="gcp-us-west1",
        ),
        metadata_probe=probe,
    )


def _seed_live_evaluation(
    settings: Settings,
    *,
    namespace: str = "pufferlab-live-doctor-target",
    regions: tuple[str, ...] = ("gcp-us-west1",),
) -> UUID:
    manifest = load_unix_dataset_manifest(UNIX_MANIFEST)
    curated = load_curated_query_manifest(CURATED_MANIFEST)
    write_spec = compile_namespace_write_spec(manifest)
    dataset = DatasetVersion(
        id=uuid4(),
        slug=manifest.slug,
        version=manifest.version,
        namespace=namespace,
        index_profile=IndexProfile(
            id=f"{manifest.slug}-{write_spec.schema_hash[:16]}",
            embedding_provider=manifest.embedding.provider,
            embedding_model=manifest.embedding.model,
            embedding_revision=manifest.embedding.revision,
            vector_attribute=manifest.vector.attribute,
            vector_dimensions=manifest.embedding.dimensions,
            vector_dtype=manifest.vector.dtype,
            distance_metric=manifest.vector.distance_metric,
            fts_profile=FtsProfile(
                tokenizer=manifest.fts.tokenizer,
                case_sensitive=manifest.fts.case_sensitive,
                language=manifest.fts.language,
                stemming=manifest.fts.stemming,
                remove_stopwords=manifest.fts.remove_stopwords,
                ascii_folding=manifest.fts.ascii_folding,
                max_token_length=manifest.fts.max_token_length,
                k1=manifest.fts.k1,
                b=manifest.fts.b,
                k3=manifest.fts.k3,
            ),
            schema_hash=write_spec.schema_hash,
        ),
        document_count=50,
        corpus_hash="doctor-live-corpus-hash",
        status=DatasetStatus.READY,
        created_at=UNIX_REVISION_CREATED_AT,
    )
    query_set_id = uuid5(
        PUFFERLAB_NAMESPACE_UUID,
        f"query-set:{dataset.id}:{curated.query_set_content_sha256}",
    )
    queries = [
        JudgedQuery(
            id=uuid5(
                PUFFERLAB_NAMESPACE_UUID,
                f"judged-query:{dataset.version}:{entry.query_id}",
            ),
            external_id=entry.query_id,
            text=f"provider-free doctor fixture {index}",
            tags=list(entry.tags),
            qrels=[
                Qrel(
                    document_id=uuid5(
                        PUFFERLAB_NAMESPACE_UUID,
                        f"doctor-document:{entry.query_id}",
                    ),
                    relevance_grade=1,
                )
            ],
        )
        for index, entry in enumerate(curated.entries)
    ]
    query_set = QuerySet(
        id=query_set_id,
        name="CQADupStack Unix curated 50",
        version=curated.selection_version,
        dataset_version_id=dataset.id,
        query_count=50,
        content_hash=curated.query_set_content_sha256 or "",
        created_at=UNIX_REVISION_CREATED_AT,
    )
    configs = derive_bound_retrieval_configs(dataset, manifest, namespace=namespace)
    with Database.from_settings(settings) as database:
        database.migrate()
        repository = PufferLabRepository(database.session_factory)
        repository.put_dataset_version(dataset)
        repository.put_query_set(query_set, queries)
        for config in configs:
            repository.put_retrieval_config(config)
        for ordinal, region in enumerate(regions):
            created_at = datetime(2026, 8, 23, tzinfo=UTC) + timedelta(seconds=ordinal)
            run = EvalRun(
                id=uuid4(),
                status=EvalRunStatus.QUEUED,
                query_set=QuerySetSummary(
                    id=query_set.id,
                    name=query_set.name,
                    version=query_set.version,
                    query_count=50,
                    content_hash=query_set.content_hash,
                ),
                baseline_config_id=configs[0].id,
                candidate_config_ids=[config.id for config in configs[1:]],
                summaries=[],
                completed_queries=0,
                total_queries=50,
                random_seed=20260822,
                environment=RunEnvironment(
                    pufferlab_git_revision="doctor-live-test",
                    turbopuffer_region=region,
                    python_version="test",
                    platform="test",
                    max_concurrency=1,
                    warmup_query_count=0,
                    query_embedding_cache_enabled=False,
                ),
                created_at=created_at,
                started_at=None,
                completed_at=None,
                error=None,
            )
            repository.create_run(run)
            repository.transition_run(
                run.id,
                EvalRunStatus.RUNNING,
                at=created_at + timedelta(milliseconds=1),
            )
            outcomes: list[QueryOutcome] = []
            for config in configs:
                for query in queries:
                    outcome = QueryOutcome(
                        run_id=run.id,
                        config_id=config.id,
                        query_id=query.id,
                        status=QueryOutcomeStatus.SUCCEEDED,
                        payload=encode_outcome_payload(
                            EvalSuccessPayload(
                                ranked_document_ids=[query.qrels[0].document_id],
                                metrics=PerQueryMetrics(
                                    ndcg_at_10=1.0,
                                    recall_at_50=1.0,
                                    mrr_at_10=1.0,
                                ),
                                total_client_wall_latency_ms=1.0,
                                stage_timings=[],
                                candidate_counts={},
                                warnings=[],
                                trace_id=uuid4(),
                            )
                        ),
                        created_at=created_at + timedelta(milliseconds=2),
                    )
                    repository.record_outcome(outcome)
                    outcomes.append(outcome)
            summaries = finalize_durable_outcomes(
                run,
                outcomes,
                query_ids=[query.id for query in queries],
            )
            repository.complete_run(
                run.id,
                summaries,
                at=created_at + timedelta(milliseconds=3),
            )
    return dataset.id


def test_demo_doctor_is_exact_provider_free_and_immutable(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _seed(settings)
    before = _snapshot(settings.database_path)
    output = io.StringIO()
    errors = io.StringIO()

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("provider/model/default database composition is forbidden")

    exit_code = main(
        ["doctor", "--mode", "demo"],
        settings_factory=lambda: settings,
        cli_application_factory=forbidden,
        stdout=output,
        stderr=errors,
    )

    assert exit_code == 0
    assert errors.getvalue() == ""
    assert output.getvalue().splitlines() == [
        "doctor mode=demo",
        (
            "check=demo state=ready "
            "dataset_version_id=0dc8c34d-7298-54da-ae8f-248394cd1cf4 "
            "datasets=1 queries=50 configs=4 completed_runs=1"
        ),
    ]
    assert _snapshot(settings.database_path) == before


def test_default_doctor_never_unwraps_secret_or_constructs_provider(tmp_path: Path) -> None:
    class ExplodingSecret(SecretStr):
        def get_secret_value(self) -> str:
            raise AssertionError("default doctor must not unwrap the key")

    settings = _settings(tmp_path / "absent").model_copy(
        update={"turbopuffer_api_key": ExplodingSecret("hostile-secret")}
    )
    output = io.StringIO()

    assert (
        main(
            ["doctor", "--mode", "live-tiny"],
            settings_factory=lambda: settings,
            stdout=output,
        )
        == 2
    )
    assert "hostile-secret" not in output.getvalue()
    assert "requirements=search_namespace,live_search_runtime" in output.getvalue()
    assert not settings.database_path.parent.exists()


def test_demo_tampering_is_action_required_and_doctor_itself_is_immutable(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data")
    _seed(settings)
    with sqlite3.connect(settings.database_path) as connection:
        row = connection.execute("SELECT payload_json FROM query_outcomes LIMIT 1").fetchone()
        assert row is not None
        payload = row[0].replace(
            "e92c3a18-f9dd-584c-b210-d84d87bb021d",
            "00000000-0000-0000-0000-000000000000",
        )
        if payload == row[0]:
            payload = row[0].replace('"ranked_document_ids":["', '"ranked_document_ids":["0')
        connection.execute(
            "UPDATE query_outcomes SET payload_json = ? WHERE rowid = "
            "(SELECT rowid FROM query_outcomes LIMIT 1)",
            (payload,),
        )
    before = _snapshot(settings.database_path)
    output = io.StringIO()

    assert (
        main(
            ["doctor", "--mode", "demo"],
            settings_factory=lambda: settings,
            stdout=output,
        )
        == 2
    )
    assert "requirements=demo_evidence" in output.getvalue()
    assert _snapshot(settings.database_path) == before


def test_evaluation_zero_multiple_and_explicit_dataset_selection(tmp_path: Path) -> None:
    empty = _settings(tmp_path / "empty")
    with Database.from_settings(empty) as database:
        database.migrate()
    before = _snapshot(empty.database_path)
    output = io.StringIO()
    assert (
        main(
            ["doctor", "--mode", "evaluation"],
            settings_factory=lambda: empty,
            stdout=output,
        )
        == 2
    )
    assert "requirements=dataset_selection" in output.getvalue()
    assert "datasets=0" in output.getvalue()
    assert _snapshot(empty.database_path) == before

    settings = _settings(tmp_path / "seeded")
    _seed(settings)
    expected = materialize_synthetic_demo()
    second = expected.dataset_version.model_copy(update={"id": uuid4()})
    with Database.from_settings(settings) as database:
        repository = PufferLabRepository(database.session_factory)
        repository.put_dataset_version(second)

    output = io.StringIO()
    assert (
        main(
            ["doctor", "--mode", "evaluation"],
            settings_factory=lambda: settings,
            stdout=output,
        )
        == 2
    )
    assert "requirements=dataset_selection" in output.getvalue()
    assert "datasets=2" in output.getvalue()

    output = io.StringIO()
    assert (
        main(
            [
                "doctor",
                "--mode",
                "evaluation",
                "--dataset-version",
                str(expected.dataset_version.id),
            ],
            settings_factory=lambda: settings,
            stdout=output,
        )
        == 0
    )
    assert "state=ready" in output.getvalue()


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE query_sets SET content_hash = 'substituted'",
        "UPDATE retrieval_configs SET config_hash = 'substituted' WHERE rowid = "
        "(SELECT rowid FROM retrieval_configs LIMIT 1)",
        "DELETE FROM query_outcomes WHERE rowid = (SELECT rowid FROM query_outcomes LIMIT 1)",
    ],
)
def test_evaluation_rejects_query_config_and_run_evidence_tampering(
    tmp_path: Path,
    statement: str,
) -> None:
    settings = _settings(tmp_path / "data")
    _seed(settings)
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(statement)
    before = _snapshot(settings.database_path)
    output = io.StringIO()

    assert (
        main(
            ["doctor", "--mode", "evaluation"],
            settings_factory=lambda: settings,
            stdout=output,
        )
        == 2
    )
    assert "state=action_required" in output.getvalue()
    assert _snapshot(settings.database_path) == before


def test_synthetic_evaluation_live_check_fails_locally_without_probe(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data", api_key="never-send")
    _seed(settings)
    probe = FakeProbe()
    dependencies = replace(default_doctor_dependencies(), metadata_probe=probe)
    output = io.StringIO()

    assert (
        main(
            ["doctor", "--mode", "evaluation", "--live"],
            settings_factory=lambda: settings,
            doctor_dependencies=dependencies,
            stdout=output,
        )
        == 2
    )
    assert probe.calls == []
    assert "requirements=live_dataset" in output.getvalue()
    assert "never-send" not in output.getvalue()


def test_live_evaluation_resolves_only_persisted_target_and_stored_region(tmp_path: Path) -> None:
    namespace = "pufferlab-persisted-live-target"
    key = "doctor-live-secret"
    settings = _settings(tmp_path / "data", api_key=key).model_copy(
        update={"pufferlab_search_namespace": "environment-target-must-not-be-used"}
    )
    dataset_id = _seed_live_evaluation(settings, namespace=namespace)
    before = _snapshot(settings.database_path)
    probe = FakeProbe(MetadataProbeState.INDEX_UP_TO_DATE)
    output = io.StringIO()

    assert (
        main(
            [
                "doctor",
                "--mode",
                "evaluation",
                "--dataset-version",
                str(dataset_id),
                "--live",
            ],
            settings_factory=lambda: settings,
            doctor_dependencies=replace(
                default_doctor_dependencies(),
                metadata_probe=probe,
            ),
            stdout=output,
        )
        == 0
    )
    assert probe.calls == [(key, "gcp-us-west1", namespace)]
    rendered = output.getvalue()
    assert key not in rendered
    assert namespace not in rendered
    assert "environment-target-must-not-be-used" not in rendered
    assert _snapshot(settings.database_path) == before


@pytest.mark.parametrize(
    ("regions", "local_region", "namespace", "requirement"),
    [
        (("gcp-us-west1",), "gcp-us-east1", "pufferlab-live", "region_match"),
        (
            ("gcp-us-west1", "gcp-us-central1"),
            "gcp-us-west1",
            "pufferlab-live",
            "region",
        ),
        (("gcp-us-west1",), "gcp-us-west1", "n" * 129, "namespace"),
    ],
)
def test_live_evaluation_rejects_region_and_namespace_ambiguity_before_probe(
    tmp_path: Path,
    regions: tuple[str, ...],
    local_region: str,
    namespace: str,
    requirement: str,
) -> None:
    settings = _settings(tmp_path / "data", api_key="never-send", region=local_region)
    _seed_live_evaluation(settings, namespace=namespace, regions=regions)
    probe = FakeProbe()
    output = io.StringIO()

    assert (
        main(
            ["doctor", "--mode", "evaluation", "--live"],
            settings_factory=lambda: settings,
            doctor_dependencies=replace(
                default_doctor_dependencies(),
                metadata_probe=probe,
            ),
            stdout=output,
        )
        == 2
    )
    assert probe.calls == []
    assert f"requirements={requirement}" in output.getvalue()
    assert "never-send" not in output.getvalue()


@pytest.mark.parametrize(
    ("state", "expected_exit"),
    [
        (MetadataProbeState.INDEX_UP_TO_DATE, 0),
        (MetadataProbeState.INDEX_UPDATING, 3),
        (MetadataProbeState.NOT_FOUND, 3),
        (MetadataProbeState.REMOTE_FAILURE, 3),
    ],
)
def test_explicit_live_tiny_probe_has_finite_states_one_call_and_redacted_output(
    tmp_path: Path,
    state: MetadataProbeState,
    expected_exit: int,
) -> None:
    marker = "pufferlab-target-hostile-marker"
    key = "api-key-hostile-marker"
    settings = _settings(tmp_path / "unused", api_key=key)
    probe = FakeProbe(state)
    dependencies = _live_tiny_dependencies(probe, namespace=marker)
    output = io.StringIO()

    exit_code = main(
        ["doctor", "--mode", "live-tiny", "--live"],
        settings_factory=lambda: settings,
        doctor_dependencies=dependencies,
        cli_application_factory=lambda _settings: pytest.fail("default composition called"),
        stdout=output,
    )

    assert exit_code == expected_exit
    assert probe.calls == [(key, "gcp-us-west1", marker)]
    rendered = output.getvalue()
    assert key not in rendered
    assert marker not in rendered
    assert str(settings.database_path) not in rendered
    assert "check=metadata" in rendered
    assert (
        "index_up_to_date=true" in rendered
        if expected_exit == 0
        else ("state=remote_failure" in rendered)
    )
    assert marker not in repr(DoctorLiveTarget(namespace=marker, region="gcp-us-west1"))
    assert marker not in repr(dependencies)


def test_probe_exception_and_cancellation_are_fixed_and_value_free(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "unused", api_key="cancel-secret")
    failure_probe = FakeProbe(failure=RuntimeError("provider-hostile-marker"))
    output = io.StringIO()
    errors = io.StringIO()
    assert (
        main(
            ["doctor", "--mode", "live-tiny", "--live"],
            settings_factory=lambda: settings,
            doctor_dependencies=_live_tiny_dependencies(failure_probe),
            stdout=output,
            stderr=errors,
        )
        == 3
    )
    assert "provider-hostile-marker" not in output.getvalue() + errors.getvalue()
    assert failure_probe.failure is not None
    assert failure_probe.failure.__traceback__ is None

    cancelled_probe = FakeProbe(failure=asyncio_cancelled_error())
    output = io.StringIO()
    errors = io.StringIO()
    assert (
        main(
            ["doctor", "--mode", "live-tiny", "--live"],
            settings_factory=lambda: settings,
            doctor_dependencies=_live_tiny_dependencies(cancelled_probe),
            stdout=output,
            stderr=errors,
        )
        == 130
    )
    assert output.getvalue() == ""
    assert errors.getvalue() == "error: doctor cancelled\n"
    assert len(cancelled_probe.calls) == 1
    assert "cancel-secret" not in errors.getvalue()
    assert cancelled_probe.failure is not None
    assert cancelled_probe.failure.__traceback__ is None


def asyncio_cancelled_error() -> BaseException:
    import asyncio

    return asyncio.CancelledError()


def test_all_mode_order_and_exit_precedence_make_at_most_one_probe(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "data", api_key="server-secret")
    _seed(settings)
    probe = FakeProbe(MetadataProbeState.INDEX_UPDATING)
    output = io.StringIO()

    assert (
        main(
            ["doctor", "--mode", "all", "--live"],
            settings_factory=lambda: settings,
            doctor_dependencies=_live_tiny_dependencies(probe),
            stdout=output,
        )
        == 3
    )
    assert len(probe.calls) == 1
    check_lines = [line for line in output.getvalue().splitlines() if line.startswith("check=")]
    assert [line.split()[0] for line in check_lines] == [
        "check=demo",
        "check=live_tiny",
        "check=evaluation",
        "check=metadata",
    ]
    assert "requirements=live_dataset" in check_lines[2]


def test_exit_precedence_puts_internal_before_remote_before_local() -> None:
    report = DoctorReport(
        mode=DoctorMode.ALL,
        checks=(
            DoctorCheck(
                name=DoctorCheckName.DEMO,
                state=DoctorCheckState.ACTION_REQUIRED,
            ),
            DoctorCheck(
                name=DoctorCheckName.METADATA,
                state=DoctorCheckState.REMOTE_FAILURE,
            ),
        ),
    )
    assert doctor_exit_code(report) == 3
    assert (
        doctor_exit_code(
            replace(
                report,
                checks=(
                    *report.checks,
                    DoctorCheck(
                        name=DoctorCheckName.INTERNAL,
                        state=DoctorCheckState.INTERNAL_ERROR,
                    ),
                ),
            )
        )
        == 1
    )


def test_report_rejects_non_allowlisted_output_values() -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        DoctorCheck(
            name=DoctorCheckName.DEMO,
            state=DoctorCheckState.ACTION_REQUIRED,
            requirements=("hostile-free-form",),  # type: ignore[arg-type]
        )


def test_parser_rejects_dataset_version_for_demo_and_live_tiny(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_settings() -> NoReturn:
        pytest.fail("settings constructed before parser rejection")

    for mode in ("demo", "live-tiny"):
        with pytest.raises(SystemExit) as error:
            main(
                ["doctor", "--mode", mode, "--dataset-version", str(UUID(int=1))],
                settings_factory=forbidden_settings,
            )
        assert error.value.code == 2
        assert "accepted only for doctor evaluation or all" in capsys.readouterr().err

    with pytest.raises(SystemExit) as error:
        main(
            ["doctor", "--mode", "demo", "--live"],
            settings_factory=forbidden_settings,
        )
    assert error.value.code == 2
    assert "accepted only for doctor live-tiny, evaluation, or all" in capsys.readouterr().err
