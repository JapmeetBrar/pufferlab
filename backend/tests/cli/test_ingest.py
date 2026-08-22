from __future__ import annotations

import io
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
        token_factory=lambda byte_count: "ab" * byte_count,
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
    calls: list[int] = []

    def token(byte_count: int) -> str:
        calls.append(byte_count)
        return "01" * byte_count

    assert resolve_owned_namespace(None, token_factory=token) == (
        "pufferlab-tiny-010101010101010101010101"
    )
    assert calls == [12]
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
