from __future__ import annotations

import asyncio
import io
import json
import shutil
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import UUID

import pytest
from pufferlab.cli.ingest import (
    IngestTinyOptions,
    TinyFixtureIngestor,
    TinyIngestionCommandError,
    resolve_owned_namespace,
)
from pufferlab.cli.main import main
from pufferlab.config import Settings
from pufferlab.datasets.ingestion import (
    EmbeddedDocument,
    IngestionReport,
    NamespaceReadiness,
)
from pufferlab.datasets.schema import NamespaceWriteSpec
from pufferlab.owned_tiny import OwnedTinyState, owned_tiny_ingest_operation

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "fixtures" / "tiny-corpus"
MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
SCHEMA_HASH = "0251f57f6166bf8f1ab8351ae0a4a797cfcf691fb0699bcfc59a4083945eea1d"


class FakeEmbedder:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(texts))
        return tuple((0.0,) * self.dimensions for _ in texts)


class FakeEmbedderFactory:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events
        self.calls: list[tuple[str, str, int, int]] = []
        self.embedders: list[FakeEmbedder] = []

    def __call__(
        self,
        *,
        model: str,
        revision: str,
        dimensions: int,
        batch_size: int,
    ) -> FakeEmbedder:
        self.events.append(("embedder_factory", model))
        self.calls.append((model, revision, dimensions, batch_size))
        embedder = FakeEmbedder(dimensions)
        self.embedders.append(embedder)
        return embedder


class FakeProvider:
    def __init__(self) -> None:
        self.close_calls = 0
        self.delete_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

    async def delete_namespace(self, namespace: str) -> None:
        del namespace
        self.delete_calls += 1
        raise AssertionError("the ingest command must never delete a namespace")


class FakeProviderFactory:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events
        self.calls: list[tuple[str, str]] = []
        self.providers: list[FakeProvider] = []

    def __call__(self, *, api_key: str, region: str) -> FakeProvider:
        self.events.append(("provider_factory", region))
        self.calls.append((api_key, region))
        provider = FakeProvider()
        self.providers.append(provider)
        return provider


class FakeWriter:
    def __init__(self) -> None:
        self.documents: dict[UUID, EmbeddedDocument] = {}
        self.specification: NamespaceWriteSpec | None = None
        self.namespaces: list[str] = []

    async def upsert_batch(
        self,
        namespace: str,
        documents: Sequence[EmbeddedDocument],
        *,
        write_spec: NamespaceWriteSpec,
    ) -> None:
        self.namespaces.append(namespace)
        self.specification = write_spec
        self.documents.update((document.id, document) for document in documents)

    async def inspect_readiness(
        self,
        namespace: str,
        *,
        expected_document_ids: frozenset[UUID],
    ) -> NamespaceReadiness:
        del namespace
        assert self.specification is not None
        return NamespaceReadiness(
            document_count=len(self.documents),
            document_ids=frozenset(self.documents),
            schema_hash=self.specification.schema_hash,
            metadata_ready=frozenset(self.documents) == expected_document_ids,
            indexes_ready=True,
        )


def _settings(*, api_key: str | None = "server-only-test-key") -> Settings:
    return Settings.model_validate(
        {
            "pufferlab_fixture_dir": FIXTURE_DIR,
            "turbopuffer_api_key": api_key,
            "turbopuffer_region": "aws-us-east-1",
        }
    )


@pytest.fixture
def isolated_owned_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path.resolve() / ".pufferlab" / "state" / "owned-tiny-v1"
    monkeypatch.setattr("pufferlab.owned_tiny._production_state_path", lambda: state)
    monkeypatch.setattr("pufferlab.owned_tiny._production_anchor_path", lambda: tmp_path.resolve())
    return state


def _ingestor(
    events: list[tuple[str, object]],
    writer: FakeWriter,
) -> tuple[TinyFixtureIngestor, FakeProviderFactory, FakeEmbedderFactory]:
    providers = FakeProviderFactory(events)
    embedders = FakeEmbedderFactory(events)
    ingestor = TinyFixtureIngestor(
        provider_factory=providers,
        embedder_factory=embedders,
        writer_factory=lambda provider: writer,
        optional_runtime_available=lambda: True,
    )
    return ingestor, providers, embedders


def test_help_documents_environment_cost_and_namespace_target(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["dataset", "ingest-tiny", "--help"])

    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "cost-bearing turbopuffer writes" in help_text
    assert "TURBOPUFFER_API_KEY" in help_text
    assert "TURBOPUFFER_REGION" in help_text
    assert "--namespace" in help_text
    assert "idempotent rerun" in " ".join(help_text.split())


def test_missing_key_exits_two_before_model_or_provider_initialization() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["dataset", "ingest-tiny"],
        settings_factory=lambda: _settings(api_key=None),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 2
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "error: TURBOPUFFER_API_KEY is required for dataset ingestion\n"


@pytest.mark.asyncio
async def test_missing_optional_runtime_fails_before_plan_or_factories() -> None:
    events: list[tuple[str, object]] = []
    writer = FakeWriter()
    providers = FakeProviderFactory(events)
    embedders = FakeEmbedderFactory(events)
    ingestor = TinyFixtureIngestor(
        provider_factory=providers,
        embedder_factory=embedders,
        writer_factory=lambda provider: writer,
        optional_runtime_available=lambda: False,
    )

    with pytest.raises(TinyIngestionCommandError) as caught:
        await ingestor.run(_settings(), IngestTinyOptions(), emit=lambda message: None)

    assert caught.value.exit_code == 2
    assert "uv sync --extra live-search" in str(caught.value)
    assert providers.calls == []
    assert embedders.calls == []
    assert events == []


@pytest.mark.asyncio
async def test_noncanonical_fixture_profile_is_rejected_before_factories(tmp_path: Path) -> None:
    fixture = tmp_path / "tiny-corpus"
    shutil.copytree(FIXTURE_DIR, fixture)
    manifest = fixture / "manifest.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(MODEL, "untrusted/model"),
        encoding="utf-8",
    )
    events: list[tuple[str, object]] = []
    writer = FakeWriter()
    ingestor, providers, embedders = _ingestor(events, writer)
    settings = Settings.model_validate(
        {
            "pufferlab_fixture_dir": fixture,
            "turbopuffer_api_key": "server-only-test-key",
        }
    )

    with pytest.raises(TinyIngestionCommandError, match="expected pinned corpus") as caught:
        await ingestor.run(settings, IngestTinyOptions(), emit=lambda message: None)

    assert caught.value.exit_code == 2
    assert providers.calls == []
    assert embedders.calls == []
    assert events == []


def test_default_namespace_is_random_owned_and_explicit_names_are_strictly_validated() -> None:
    generated = resolve_owned_namespace(None)
    assert generated.startswith("pufferlab-tiny-")
    assert len(generated) == len("pufferlab-tiny-") + 24
    assert all(character in "0123456789abcdef" for character in generated[-24:])
    assert resolve_owned_namespace("pufferlab-my.safe_namespace-1") == (
        "pufferlab-my.safe_namespace-1"
    )


@pytest.mark.parametrize(
    "namespace",
    [
        "",
        "foreign-namespace",
        "pufferlab-",
        "pufferlab-/unsafe",
        f"pufferlab-{'a' * 119}",
    ],
)
def test_explicit_namespace_rejects_unowned_or_unsafe_values(namespace: str) -> None:
    with pytest.raises(TinyIngestionCommandError) as caught:
        resolve_owned_namespace(namespace)

    assert caught.value.exit_code == 2


@pytest.mark.asyncio
async def test_manifest_plan_precedes_factories_and_success_prints_exact_search_namespace() -> None:
    events: list[tuple[str, object]] = []
    writer = FakeWriter()
    ingestor, providers, embedders = _ingestor(events, writer)

    def emit(message: str) -> None:
        events.append(("output", message))

    report = await ingestor.run(
        _settings(),
        IngestTinyOptions(namespace="pufferlab-repeatable", batch_size=7),
        emit=emit,
    )

    assert report.ready
    assert providers.calls == [("server-only-test-key", "aws-us-east-1")]
    assert embedders.calls == [(MODEL, REVISION, 384, 7)]
    first_factory = next(
        index for index, event in enumerate(events) if event[0].endswith("factory")
    )
    plan_output = [value for kind, value in events[:first_factory] if kind == "output"]
    assert plan_output == [
        "ingestion plan (local model execution and remote writes follow)",
        "region=aws-us-east-1",
        "namespace=pufferlab-repeatable",
        f"schema_hash={SCHEMA_HASH}",
        "documents=20",
        f"embedding_model={MODEL} revision={REVISION} dimensions=384",
    ]
    output = [value for kind, value in events if kind == "output"]
    assert "progress state=ingesting batches=0/3 documents=0/20" in output
    assert "progress state=verifying batches=3/3 documents=20/20" in output
    assert output[-3] == (
        f"verified remote_documents=20 exact_document_ids=true "
        f"observed_schema_hash={SCHEMA_HASH} distance_metric=cosine_distance "
        "metadata_ready=true indexes_ready=true"
    )
    assert output[-2] == (
        f"ready namespace=pufferlab-repeatable documents=20 schema_hash={SCHEMA_HASH}"
    )
    assert output[-1] == "PUFFERLAB_SEARCH_NAMESPACE=pufferlab-repeatable"
    assert "server-only-test-key" not in "\n".join(output)
    assert all(provider.close_calls == 1 for provider in providers.providers)
    assert all(provider.delete_calls == 0 for provider in providers.providers)


@pytest.mark.asyncio
async def test_same_explicit_namespace_rerun_is_idempotent_and_never_deletes() -> None:
    events: list[tuple[str, object]] = []
    writer = FakeWriter()
    ingestor, providers, _ = _ingestor(events, writer)
    options = IngestTinyOptions(namespace="pufferlab-idempotent", batch_size=5)

    first = await ingestor.run(_settings(), options, emit=lambda message: None)
    first_documents = writer.documents.copy()
    second = await ingestor.run(_settings(), options, emit=lambda message: None)

    assert first.ready and second.ready
    assert writer.documents == first_documents
    assert len(writer.documents) == 20
    assert set(writer.namespaces) == {"pufferlab-idempotent"}
    assert len(writer.namespaces) == 8
    assert len(providers.providers) == 2
    assert all(provider.close_calls == 1 for provider in providers.providers)
    assert all(provider.delete_calls == 0 for provider in providers.providers)


def test_main_redacts_unexpected_runner_failure_and_returns_one() -> None:
    secret = "provider-secret-that-must-not-leak"

    async def failing_runner(
        settings: Settings,
        options: IngestTinyOptions,
        *,
        emit: Callable[[str], None],
    ) -> IngestionReport:
        del settings, options, emit
        raise RuntimeError(secret)

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(
        ["dataset", "ingest-tiny", "--namespace", "pufferlab-safe"],
        settings_factory=_settings,
        ingest_runner=failing_runner,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "error: tiny fixture ingestion failed\n"
    assert secret not in stderr.getvalue()


@pytest.mark.asyncio
async def test_runtime_failure_is_detached_and_redacts_factory_details() -> None:
    secret = "embedder-factory-secret"

    def failing_embedder(**kwargs: object) -> FakeEmbedder:
        del kwargs
        raise RuntimeError(secret)

    ingestor = TinyFixtureIngestor(
        embedder_factory=failing_embedder,
        optional_runtime_available=lambda: True,
    )

    with pytest.raises(TinyIngestionCommandError) as caught:
        await ingestor.run(
            _settings(),
            IngestTinyOptions(namespace="pufferlab-safe"),
            emit=lambda message: None,
        )

    formatted = "".join(traceback.format_exception(caught.value, chain=True))
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert secret not in repr(caught.value)
    assert secret not in formatted


@pytest.mark.asyncio
async def test_validation_failure_does_not_retain_api_key_in_ingest_traceback_locals() -> None:
    secret = "validation-retained-api-key-marker"
    ingestor = TinyFixtureIngestor(optional_runtime_available=lambda: False)

    with pytest.raises(TinyIngestionCommandError) as caught:
        await ingestor.run(
            _settings(api_key=secret),
            IngestTinyOptions(),
            emit=lambda message: None,
        )

    traceback_value = caught.value.__traceback__
    production_locals: list[str] = []
    while traceback_value is not None:
        if traceback_value.tb_frame.f_code.co_filename.endswith("/pufferlab/cli/ingest.py"):
            production_locals.append(repr(traceback_value.tb_frame.f_locals))
        traceback_value = traceback_value.tb_next
    assert secret not in "".join(production_locals)


@pytest.mark.asyncio
async def test_generated_ingest_persists_intent_before_factories_and_reaches_ready(
    isolated_owned_state: Path,
) -> None:
    events: list[tuple[str, object]] = []
    writer = FakeWriter()
    providers = FakeProviderFactory(events)
    embedders = FakeEmbedderFactory(events)

    class InspectingProviderFactory:
        def __call__(self, *, api_key: str, region: str) -> FakeProvider:
            receipt = json.loads(
                (isolated_owned_state / "receipt.json").read_text(encoding="utf-8")
            )
            assert receipt["state"] == "intent"
            assert (isolated_owned_state / "owner.key").is_file()
            assert (isolated_owned_state / "operation.lock").is_file()
            return providers(api_key=api_key, region=region)

    ingestor = TinyFixtureIngestor(
        provider_factory=InspectingProviderFactory(),
        embedder_factory=embedders,
        writer_factory=lambda provider: writer,
        optional_runtime_available=lambda: True,
    )

    report = await ingestor.run(_settings(), IngestTinyOptions(), emit=lambda message: None)
    receipt = json.loads((isolated_owned_state / "receipt.json").read_text(encoding="utf-8"))

    assert report.ready
    assert report.namespace == receipt["namespace"]
    assert receipt["state"] == "ready"
    assert providers.calls == [("server-only-test-key", "aws-us-east-1")]
    assert set(writer.namespaces) == {report.namespace}


@pytest.mark.asyncio
async def test_initial_receipt_publish_collision_fails_before_model_or_provider(
    isolated_owned_state: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlab import owned_tiny

    events: list[tuple[str, object]] = []
    ingestor, providers, embedders = _ingestor(events, FakeWriter())
    real_rename_noreplace = owned_tiny._rename_noreplace
    collision_inode: int | None = None

    def collide_before_receipt_publish(
        directory_fd: int,
        source: str,
        destination: str,
    ) -> None:
        nonlocal collision_inode
        if destination == "receipt.json" and collision_inode is None:
            staged = isolated_owned_state / source
            collision = isolated_owned_state / "receipt-publish-collider"
            collision.write_bytes(staged.read_bytes())
            collision.chmod(0o600)
            collision_inode = collision.stat().st_ino
            collision.replace(isolated_owned_state / "receipt.json")
        real_rename_noreplace(directory_fd, source, destination)

    monkeypatch.setattr(owned_tiny, "_rename_noreplace", collide_before_receipt_publish)
    with pytest.raises(TinyIngestionCommandError, match="intent could not be persisted"):
        await ingestor.run(_settings(), IngestTinyOptions(), emit=lambda message: None)

    assert collision_inode is not None
    assert (isolated_owned_state / "receipt.json").stat().st_ino == collision_inode
    assert providers.calls == []
    assert embedders.calls == []


@pytest.mark.asyncio
async def test_writable_relocated_state_cannot_mint_second_receipt_or_start_factories(
    isolated_owned_state: Path,
) -> None:
    events: list[tuple[str, object]] = []
    ingestor, providers, embedders = _ingestor(events, FakeWriter())
    pufferlab_directory = isolated_owned_state.parents[1]
    relocated = pufferlab_directory.parent / ".pufferlab-authenticated-relocated"

    with owned_tiny_ingest_operation() as operation:
        snapshot = operation.create_intent(
            api_key="server-only-test-key",
            region="aws-us-east-1",
        )
        pufferlab_directory.replace(relocated)
        pufferlab_directory.mkdir(mode=0o777)
        pufferlab_directory.chmod(0o777)

        with pytest.raises(TinyIngestionCommandError):
            await ingestor.run(_settings(), IngestTinyOptions(), emit=lambda message: None)

    with pytest.raises(TinyIngestionCommandError):
        await ingestor.run(_settings(), IngestTinyOptions(), emit=lambda message: None)

    receipts = list(pufferlab_directory.parent.rglob("receipt.json"))
    assert receipts == [relocated / "state/owned-tiny-v1/receipt.json"]
    assert receipts[0].read_bytes() == snapshot.raw
    assert providers.calls == []
    assert embedders.calls == []
    assert events == []


@pytest.mark.asyncio
async def test_generated_ingest_resumes_exact_receipt_and_uses_creating_region(
    isolated_owned_state: Path,
) -> None:
    events: list[tuple[str, object]] = []
    writer = FakeWriter()
    ingestor, providers, _ = _ingestor(events, writer)
    first = await ingestor.run(_settings(), IngestTinyOptions(), emit=lambda message: None)
    original = (isolated_owned_state / "receipt.json").read_bytes()
    changed_region = Settings.model_validate(
        {
            "pufferlab_fixture_dir": FIXTURE_DIR,
            "turbopuffer_api_key": "server-only-test-key",
            "turbopuffer_region": "gcp-europe-west3",
        }
    )

    second = await ingestor.run(changed_region, IngestTinyOptions(), emit=lambda message: None)

    assert second.namespace == first.namespace
    assert (isolated_owned_state / "receipt.json").read_bytes() == original
    assert providers.calls == [
        ("server-only-test-key", "aws-us-east-1"),
        ("server-only-test-key", "aws-us-east-1"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("attack", ["replace-lock", "relocate-state"])
async def test_anchor_rejects_split_operation_before_model_or_provider(
    isolated_owned_state: Path,
    attack: str,
) -> None:
    ingestor, providers, embedders = _ingestor([], FakeWriter())

    with owned_tiny_ingest_operation() as first_operation:
        first_operation.create_intent(
            api_key="server-only-test-key",
            region="aws-us-east-1",
        )
        if attack == "replace-lock":
            replacement = isolated_owned_state / "replacement-lock"
            replacement.write_bytes(b"")
            replacement.chmod(0o600)
            replacement.replace(isolated_owned_state / "operation.lock")
        else:
            isolated_owned_state.replace(isolated_owned_state.parent / "relocated-state")

        with pytest.raises(TinyIngestionCommandError, match="already running"):
            await ingestor.run(_settings(), IngestTinyOptions(), emit=lambda message: None)

    assert providers.calls == []
    assert embedders.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("attack", ["replace-lock", "relocate-state"])
async def test_generated_ingest_rechecks_continuity_before_next_writer_action(
    isolated_owned_state: Path,
    attack: str,
) -> None:
    with owned_tiny_ingest_operation() as operation:
        snapshot = operation.create_intent(
            api_key="server-only-test-key",
            region="aws-us-east-1",
        )
        snapshot = operation.transition(snapshot, OwnedTinyState.CREATED)
        operation.transition(snapshot, OwnedTinyState.READY)

    class AttackingWriter(FakeWriter):
        def __init__(self) -> None:
            super().__init__()
            self.readiness_calls = 0

        async def upsert_batch(
            self,
            namespace: str,
            documents: Sequence[EmbeddedDocument],
            *,
            write_spec: NamespaceWriteSpec,
        ) -> None:
            await super().upsert_batch(namespace, documents, write_spec=write_spec)
            if attack == "replace-lock":
                replacement = isolated_owned_state / "replacement-lock"
                replacement.write_bytes(b"")
                replacement.chmod(0o600)
                replacement.replace(isolated_owned_state / "operation.lock")
            else:
                isolated_owned_state.replace(isolated_owned_state.parent / "relocated-state")

        async def inspect_readiness(
            self,
            namespace: str,
            *,
            expected_document_ids: frozenset[UUID],
        ) -> NamespaceReadiness:
            self.readiness_calls += 1
            return await super().inspect_readiness(
                namespace,
                expected_document_ids=expected_document_ids,
            )

    writer = AttackingWriter()
    ingestor, providers, _ = _ingestor([], writer)

    with pytest.raises(TinyIngestionCommandError, match="before readiness"):
        await ingestor.run(
            _settings(),
            IngestTinyOptions(readiness_attempts=1, readiness_poll_interval=0),
            emit=lambda message: None,
        )

    assert len(writer.namespaces) == 1
    assert writer.readiness_calls == 0
    assert len(providers.calls) == 1
    assert providers.providers[0].close_calls == 1


@pytest.mark.asyncio
async def test_ambiguous_generated_factory_failure_retains_intent_and_rerun_target(
    isolated_owned_state: Path,
) -> None:
    marker = "private-provider-start-marker"

    def fail_provider(**kwargs: object) -> FakeProvider:
        del kwargs
        raise RuntimeError(marker)

    first_ingestor = TinyFixtureIngestor(
        provider_factory=fail_provider,
        embedder_factory=FakeEmbedderFactory([]),
        writer_factory=lambda provider: FakeWriter(),
        optional_runtime_available=lambda: True,
    )
    with pytest.raises(TinyIngestionCommandError) as caught:
        await first_ingestor.run(_settings(), IngestTinyOptions(), emit=lambda message: None)
    first_receipt = json.loads((isolated_owned_state / "receipt.json").read_text(encoding="utf-8"))

    writer = FakeWriter()
    second_ingestor, providers, _ = _ingestor([], writer)
    report = await second_ingestor.run(
        _settings(),
        IngestTinyOptions(),
        emit=lambda message: None,
    )

    assert marker not in str(caught.value)
    assert first_receipt["state"] == "intent"
    assert report.namespace == first_receipt["namespace"]
    assert providers.calls == [("server-only-test-key", "aws-us-east-1")]


@pytest.mark.asyncio
async def test_generated_provider_close_failure_retains_created_receipt(
    isolated_owned_state: Path,
) -> None:
    class CloseFailingProvider(FakeProvider):
        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("private-close-marker")

    events: list[tuple[str, object]] = []

    def provider_factory(*, api_key: str, region: str) -> CloseFailingProvider:
        del api_key, region
        return CloseFailingProvider()

    ingestor = TinyFixtureIngestor(
        provider_factory=provider_factory,
        embedder_factory=FakeEmbedderFactory(events),
        writer_factory=lambda provider: FakeWriter(),
        optional_runtime_available=lambda: True,
    )

    with pytest.raises(TinyIngestionCommandError, match="close cleanly"):
        await ingestor.run(_settings(), IngestTinyOptions(), emit=lambda message: None)

    receipt = json.loads((isolated_owned_state / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["state"] == "created"


@pytest.mark.asyncio
async def test_rotated_key_resume_fails_before_factories_and_retains_ready_receipt(
    isolated_owned_state: Path,
) -> None:
    writer = FakeWriter()
    ingestor, providers, embedders = _ingestor([], writer)
    await ingestor.run(_settings(), IngestTinyOptions(), emit=lambda message: None)
    original = (isolated_owned_state / "receipt.json").read_bytes()

    with pytest.raises(TinyIngestionCommandError, match="does not match") as caught:
        await ingestor.run(
            _settings(api_key="rotated-key"),
            IngestTinyOptions(),
            emit=lambda message: None,
        )

    assert caught.value.exit_code == 2
    assert len(providers.calls) == 1
    assert len(embedders.calls) == 1
    assert (isolated_owned_state / "receipt.json").read_bytes() == original


@pytest.mark.asyncio
async def test_receipt_replacement_during_model_setup_fails_before_provider(
    isolated_owned_state: Path,
) -> None:
    events: list[tuple[str, object]] = []
    providers = FakeProviderFactory(events)
    delegate = FakeEmbedderFactory(events)

    def replacing_embedder(
        *,
        model: str,
        revision: str,
        dimensions: int,
        batch_size: int,
    ) -> FakeEmbedder:
        receipt = isolated_owned_state / "receipt.json"
        raw = receipt.read_bytes()
        replacement = isolated_owned_state / "replacement"
        replacement.write_bytes(raw)
        replacement.chmod(0o600)
        replacement.replace(receipt)
        return delegate(
            model=model,
            revision=revision,
            dimensions=dimensions,
            batch_size=batch_size,
        )

    ingestor = TinyFixtureIngestor(
        provider_factory=providers,
        embedder_factory=replacing_embedder,
        writer_factory=lambda provider: FakeWriter(),
        optional_runtime_available=lambda: True,
    )

    with pytest.raises(TinyIngestionCommandError, match="runtime failed"):
        await ingestor.run(_settings(), IngestTinyOptions(), emit=lambda message: None)

    assert providers.calls == []
    receipt_payload = json.loads(
        (isolated_owned_state / "receipt.json").read_text(encoding="utf-8")
    )
    assert receipt_payload["state"] == "intent"


@pytest.mark.asyncio
async def test_explicit_namespace_never_creates_or_reads_owned_state(
    isolated_owned_state: Path,
) -> None:
    events: list[tuple[str, object]] = []
    writer = FakeWriter()
    ingestor, _, _ = _ingestor(events, writer)

    report = await ingestor.run(
        _settings(),
        IngestTinyOptions(namespace="pufferlab-explicit-caller-owned"),
        emit=lambda message: None,
    )

    assert report.namespace == "pufferlab-explicit-caller-owned"
    assert not isolated_owned_state.exists()


def test_generated_cancellation_closes_provider_and_retains_intent(
    isolated_owned_state: Path,
) -> None:
    class CancellingWriter(FakeWriter):
        async def upsert_batch(
            self,
            namespace: str,
            documents: Sequence[EmbeddedDocument],
            *,
            write_spec: NamespaceWriteSpec,
        ) -> None:
            del namespace, documents, write_spec
            raise asyncio.CancelledError()

    events: list[tuple[str, object]] = []
    provider_factory = FakeProviderFactory(events)
    ingestor = TinyFixtureIngestor(
        provider_factory=provider_factory,
        embedder_factory=FakeEmbedderFactory(events),
        writer_factory=lambda provider: CancellingWriter(),
        optional_runtime_available=lambda: True,
    )
    stderr = io.StringIO()

    exit_code = main(
        ["dataset", "ingest-tiny"],
        settings_factory=_settings,
        ingest_runner=ingestor.run,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    receipt = json.loads((isolated_owned_state / "receipt.json").read_text(encoding="utf-8"))
    assert exit_code == 130
    assert receipt["state"] == "intent"
    assert provider_factory.providers[0].close_calls == 1
    assert stderr.getvalue() == "error: tiny fixture ingestion cancelled\n"


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize(
    ("stage", "expected_state", "provider_constructed"),
    [
        ("embedder", "intent", False),
        ("provider", "intent", False),
        ("writer", "intent", True),
        ("upsert", "intent", True),
        ("readiness", "created", True),
        ("close", "created", True),
    ],
)
def test_generated_process_control_drains_provider_and_retains_resumable_receipt(
    isolated_owned_state: Path,
    control: type[BaseException],
    stage: str,
    expected_state: str,
    provider_constructed: bool,
) -> None:
    marker = f"private-ingest-{stage}-{control.__name__}-marker"
    providers: list[FakeProvider] = []

    class ControlledProvider(FakeProvider):
        async def close(self) -> None:
            self.close_calls += 1
            if stage == "close":
                raise control(marker)

    class ControlledWriter(FakeWriter):
        async def upsert_batch(
            self,
            namespace: str,
            documents: Sequence[EmbeddedDocument],
            *,
            write_spec: NamespaceWriteSpec,
        ) -> None:
            if stage == "upsert":
                raise control(marker)
            await super().upsert_batch(namespace, documents, write_spec=write_spec)

        async def inspect_readiness(
            self,
            namespace: str,
            *,
            expected_document_ids: frozenset[UUID],
        ) -> NamespaceReadiness:
            if stage == "readiness":
                raise control(marker)
            return await super().inspect_readiness(
                namespace,
                expected_document_ids=expected_document_ids,
            )

    def provider_factory(*, api_key: str, region: str) -> ControlledProvider:
        del api_key, region
        if stage == "provider":
            raise control(marker)
        provider = ControlledProvider()
        providers.append(provider)
        return provider

    def embedder_factory(**kwargs: object) -> FakeEmbedder:
        if stage == "embedder":
            raise control(marker)
        return FakeEmbedder(int(kwargs["dimensions"]))

    def writer_factory(provider: FakeProvider) -> ControlledWriter:
        del provider
        if stage == "writer":
            raise control(marker)
        return ControlledWriter()

    ingestor = TinyFixtureIngestor(
        provider_factory=provider_factory,
        embedder_factory=embedder_factory,
        writer_factory=writer_factory,
        optional_runtime_available=lambda: True,
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["dataset", "ingest-tiny"],
        settings_factory=_settings,
        ingest_runner=ingestor.run,
        stdout=stdout,
        stderr=stderr,
    )

    receipt = json.loads((isolated_owned_state / "receipt.json").read_text(encoding="utf-8"))
    assert exit_code == (130 if control is KeyboardInterrupt else 1)
    assert receipt["state"] == expected_state
    assert len(providers) == int(provider_constructed)
    assert sum(provider.close_calls for provider in providers) == int(provider_constructed)
    assert marker not in stdout.getvalue() + stderr.getvalue()
    assert "server-only-test-key" not in stdout.getvalue() + stderr.getvalue()


@pytest.mark.asyncio
async def test_generated_process_control_trace_retains_no_key_target_provider_or_writer(
    isolated_owned_state: Path,
) -> None:
    marker = "private-ingest-process-control-graph-marker"

    class MarkedProvider(FakeProvider):
        def __repr__(self) -> str:
            return marker

    class MarkedWriter(FakeWriter):
        def __repr__(self) -> str:
            return marker

    provider = MarkedProvider()

    def writer_factory(created_provider: FakeProvider) -> MarkedWriter:
        assert created_provider is provider
        raise SystemExit(marker)

    ingestor = TinyFixtureIngestor(
        provider_factory=lambda **kwargs: provider,
        embedder_factory=FakeEmbedderFactory([]),
        writer_factory=writer_factory,
        optional_runtime_available=lambda: True,
    )

    with pytest.raises(TinyIngestionCommandError) as caught:
        await ingestor.run(_settings(), IngestTinyOptions(), emit=lambda message: None)

    receipt = json.loads((isolated_owned_state / "receipt.json").read_text(encoding="utf-8"))
    production_locals: list[str] = []
    traceback_value = caught.value.__traceback__
    while traceback_value is not None:
        if traceback_value.tb_frame.f_code.co_filename.endswith("/pufferlab/cli/ingest.py"):
            production_locals.append(repr(traceback_value.tb_frame.f_locals))
        traceback_value = traceback_value.tb_next
    rendered = "".join(production_locals)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert marker not in rendered
    assert "server-only-test-key" not in rendered
    assert receipt["namespace"] not in rendered
    assert provider.close_calls == 1
    assert receipt["state"] == "intent"
