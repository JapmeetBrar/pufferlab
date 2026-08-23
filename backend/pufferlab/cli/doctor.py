"""Provider-free local diagnostics with an explicit one-shot live option."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID, uuid5

from pufferlab.application.evaluation_views import EvaluationViewService
from pufferlab.application.readiness import LocalCapabilityInspector
from pufferlab.config import Settings
from pufferlab.contracts.capabilities import (
    CapabilitiesResponse,
    CapabilityActionCode,
    CapabilityRequirementCode,
    CapabilityState,
)
from pufferlab.contracts.datasets import DataOrigin, DatasetStatus, DatasetVersion
from pufferlab.contracts.evals import EvalRun, EvalRunStatus, QuerySet
from pufferlab.contracts.retrieval import RetrievalConfig
from pufferlab.datasets.cqadupstack import load_curated_query_manifest
from pufferlab.datasets.identity import PUFFERLAB_NAMESPACE_UUID
from pufferlab.datasets.unix_application import UNIX_REVISION_CREATED_AT
from pufferlab.persistence.read_only import (
    ExistingReadOnlyCatalog,
    open_existing_read_only_catalog,
)
from pufferlab.persistence.repository import PufferLabRepository
from pufferlab.providers.metadata_probe import (
    MetadataProbeConfigurationError,
    MetadataProbeResult,
    MetadataProbeState,
    is_valid_metadata_probe_region,
    probe_namespace_metadata,
)
from pufferlab.retrieval.config import derive_bound_retrieval_configs
from pufferlab.synthetic_demo.authored import AUTHORED_SYNTHETIC_DEMO
from pufferlab.synthetic_demo.seeder import materialize_synthetic_demo

_ROOT = Path(__file__).resolve().parents[3]
_UNIX_MANIFEST = _ROOT / "datasets" / "cqadupstack-unix" / "dataset-manifest.json"
_CURATED_MANIFEST = _ROOT / "datasets" / "cqadupstack-unix" / "curated-50.json"
_MAX_NAMESPACE_BYTES = 128


class DoctorMode(StrEnum):
    DEMO = "demo"
    LIVE_TINY = "live-tiny"
    EVALUATION = "evaluation"
    ALL = "all"


class DoctorCheckName(StrEnum):
    DEMO = "demo"
    LIVE_TINY = "live_tiny"
    EVALUATION = "evaluation"
    METADATA = "metadata"
    INTERNAL = "internal"


class DoctorCheckState(StrEnum):
    READY = "ready"
    ACTION_REQUIRED = "action_required"
    REMOTE_FAILURE = "remote_failure"
    INTERNAL_ERROR = "internal_error"


class DoctorRequirementCode(StrEnum):
    CATALOG = "catalog"
    DEMO_EVIDENCE = "demo_evidence"
    DATASET_SELECTION = "dataset_selection"
    DATASET_READY = "dataset_ready"
    QUERY_SET = "query_set"
    CONFIG_CATALOG = "config_catalog"
    COMPLETED_RUN = "completed_run"
    LIVE_DATASET = "live_dataset"
    NAMESPACE = "namespace"
    REGION = "region"
    REGION_MATCH = "region_match"
    API_KEY = "api_key"
    OWNED_TINY_RECEIPT = "owned_tiny_receipt"


class DoctorActionCode(StrEnum):
    SEED_DEMO = "seed_demo"
    SELECT_DATASET = "select_dataset"
    REPAIR_CATALOG = "repair_catalog"
    COMPLETE_EVALUATION = "complete_evaluation"
    SELECT_LIVE_DATASET = "select_live_dataset"
    CONFIGURE_API_KEY = "configure_api_key"
    CONFIGURE_REGION = "configure_region"
    RESOLVE_OWNED_TINY_RECEIPT = "resolve_owned_tiny_receipt"


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: DoctorCheckName
    state: DoctorCheckState
    requirements: tuple[DoctorRequirementCode | CapabilityRequirementCode, ...] = ()
    next_action: DoctorActionCode | CapabilityActionCode | None = None
    dataset_version_id: UUID | None = None
    dataset_count: int | None = None
    query_count: int | None = None
    config_count: int | None = None
    completed_run_count: int | None = None
    metadata_reachable: bool | None = None
    index_up_to_date: bool | None = None
    index_updating: bool | None = None

    def __post_init__(self) -> None:
        finite_requirements = (DoctorRequirementCode, CapabilityRequirementCode)
        finite_actions = (DoctorActionCode, CapabilityActionCode)
        if not isinstance(self.name, DoctorCheckName) or not isinstance(
            self.state, DoctorCheckState
        ):
            raise ValueError("doctor check name and state must be allowlisted")
        if any(not isinstance(item, finite_requirements) for item in self.requirements):
            raise ValueError("doctor requirements must be allowlisted")
        if self.next_action is not None and not isinstance(self.next_action, finite_actions):
            raise ValueError("doctor action must be allowlisted")
        if self.dataset_version_id is not None and not isinstance(self.dataset_version_id, UUID):
            raise ValueError("doctor dataset identity must be a UUID")
        metadata_flags = (
            self.metadata_reachable,
            self.index_up_to_date,
            self.index_updating,
        )
        if any(value is not None and not isinstance(value, bool) for value in metadata_flags):
            raise ValueError("doctor metadata state must be boolean")
        bounds = (
            (self.dataset_count, 2),
            (self.query_count, 50),
            (self.config_count, 4),
            (self.completed_run_count, 100),
        )
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum)
            for value, maximum in bounds
        ):
            raise ValueError("doctor report count is outside its bound")


@dataclass(frozen=True, slots=True)
class DoctorReport:
    mode: DoctorMode
    checks: tuple[DoctorCheck, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mode, DoctorMode):
            raise ValueError("doctor mode must be allowlisted")
        names = [check.name for check in self.checks]
        order = tuple(DoctorCheckName)
        if (
            not self.checks
            or len(self.checks) > 4
            or len(names) != len(set(names))
            or names != sorted(names, key=order.index)
        ):
            raise ValueError("doctor checks must be unique and use the frozen order")


@dataclass(frozen=True, slots=True)
class DoctorExecution:
    report: DoctorReport
    exit_code: int

    def __post_init__(self) -> None:
        if self.exit_code not in {0, 1, 2, 3, 130}:
            raise ValueError("doctor exit code must be allowlisted")


@dataclass(frozen=True, slots=True, repr=False)
class DoctorLiveTarget:
    namespace: str = field(repr=False)
    region: str = field(repr=False)


class CapabilityInspection(Protocol):
    def inspect(self) -> CapabilitiesResponse: ...


class CatalogFactory(Protocol):
    def __call__(self, path: Path) -> ExistingReadOnlyCatalog: ...


class OwnedTinyTargetResolver(Protocol):
    def __call__(self, settings: Settings) -> DoctorLiveTarget | None: ...


class MetadataProbe(Protocol):
    async def __call__(
        self,
        *,
        api_key: str,
        region: str,
        namespace: str,
    ) -> MetadataProbeResult: ...


@dataclass(frozen=True, slots=True, repr=False)
class DoctorDependencies:
    capability_inspector_factory: Callable[[Settings], CapabilityInspection]
    catalog_factory: CatalogFactory
    owned_tiny_target_resolver: OwnedTinyTargetResolver
    metadata_probe: MetadataProbe


class _ProbeControl(StrEnum):
    NONE = "none"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class _ExplicitProbeOutcome:
    check: DoctorCheck | None = None
    control: _ProbeControl = _ProbeControl.NONE


def default_doctor_dependencies() -> DoctorDependencies:
    return DoctorDependencies(
        capability_inspector_factory=LocalCapabilityInspector,
        catalog_factory=open_existing_read_only_catalog,
        owned_tiny_target_resolver=lambda _settings: None,
        metadata_probe=probe_namespace_metadata,
    )


async def run_doctor(
    settings: Settings,
    *,
    mode: DoctorMode,
    dataset_version_id: UUID | None,
    live: bool,
    dependencies: DoctorDependencies | None = None,
) -> DoctorExecution:
    """Run independent local checks, followed by at most one explicit metadata request."""

    dependencies = dependencies or default_doctor_dependencies()
    checks: list[DoctorCheck] = []
    live_tiny_target: DoctorLiveTarget | None = None
    evaluation_target: DoctorLiveTarget | None = None
    catalog: ExistingReadOnlyCatalog | None = None
    catalog_needed = mode in {DoctorMode.DEMO, DoctorMode.EVALUATION, DoctorMode.ALL}

    try:
        if catalog_needed:
            try:
                catalog = dependencies.catalog_factory(settings.database_path)
            except Exception as error:
                _detach_exception(error)
                catalog = None

        if mode in {DoctorMode.DEMO, DoctorMode.ALL}:
            checks.append(_inspect_demo(catalog))

        if mode in {DoctorMode.LIVE_TINY, DoctorMode.ALL}:
            live_tiny_check, live_tiny_target = _inspect_live_tiny(settings, dependencies)
            checks.append(live_tiny_check)

        if mode in {DoctorMode.EVALUATION, DoctorMode.ALL}:
            evaluation_check, evaluation_target = _inspect_evaluation(
                catalog,
                dataset_version_id=dataset_version_id,
                require_live=live,
                local_region=settings.turbopuffer_region,
            )
            checks.append(evaluation_check)
    except Exception as error:
        _detach_exception(error)
        checks.append(
            DoctorCheck(
                name=DoctorCheckName.INTERNAL,
                state=DoctorCheckState.INTERNAL_ERROR,
            )
        )
    finally:
        if catalog is not None:
            try:
                catalog.close()
            except Exception as error:
                _detach_exception(error)
                evaluation_target = None
                checks = [
                    DoctorCheck(
                        name=check.name,
                        state=DoctorCheckState.ACTION_REQUIRED,
                        requirements=(DoctorRequirementCode.CATALOG,),
                        next_action=DoctorActionCode.REPAIR_CATALOG,
                    )
                    if check.name in {DoctorCheckName.DEMO, DoctorCheckName.EVALUATION}
                    else check
                    for check in checks
                ]

    if live and not any(check.state is DoctorCheckState.INTERNAL_ERROR for check in checks):
        target = evaluation_target or live_tiny_target
        if target is not None:
            probe_outcome = await _probe_explicit_target(
                settings,
                target=target,
                probe=dependencies.metadata_probe,
            )
            if probe_outcome.control is _ProbeControl.CANCELLED:
                return DoctorExecution(
                    report=DoctorReport(mode=mode, checks=tuple(checks)),
                    exit_code=130,
                )
            if probe_outcome.control is _ProbeControl.INTERNAL:
                checks.append(
                    DoctorCheck(
                        name=DoctorCheckName.INTERNAL,
                        state=DoctorCheckState.INTERNAL_ERROR,
                    )
                )
            elif probe_outcome.check is not None:
                checks.append(probe_outcome.check)

    report = DoctorReport(mode=mode, checks=tuple(checks))
    return DoctorExecution(report=report, exit_code=doctor_exit_code(report))


def doctor_exit_code(report: DoctorReport) -> int:
    if any(check.state is DoctorCheckState.INTERNAL_ERROR for check in report.checks):
        return 1
    if any(check.state is DoctorCheckState.REMOTE_FAILURE for check in report.checks):
        return 3
    if any(check.state is DoctorCheckState.ACTION_REQUIRED for check in report.checks):
        return 2
    return 0


def render_doctor(report: DoctorReport) -> tuple[str, ...]:
    lines = [f"doctor mode={report.mode.value}"]
    for check in report.checks:
        fields = [f"check={check.name.value}", f"state={check.state.value}"]
        if check.requirements:
            fields.append(f"requirements={','.join(item.value for item in check.requirements)}")
        if check.next_action is not None:
            fields.append(f"next_action={check.next_action.value}")
        if check.dataset_version_id is not None:
            fields.append(f"dataset_version_id={check.dataset_version_id}")
        for name, value in (
            ("datasets", check.dataset_count),
            ("queries", check.query_count),
            ("configs", check.config_count),
            ("completed_runs", check.completed_run_count),
        ):
            if value is not None:
                fields.append(f"{name}={value}")
        for name, value in (
            ("metadata_reachable", check.metadata_reachable),
            ("index_up_to_date", check.index_up_to_date),
            ("index_updating", check.index_updating),
        ):
            if value is not None:
                fields.append(f"{name}={str(value).lower()}")
        lines.append(" ".join(fields))
    return tuple(lines)


def _inspect_demo(catalog: ExistingReadOnlyCatalog | None) -> DoctorCheck:
    if catalog is None:
        return _local_failure(
            DoctorCheckName.DEMO,
            DoctorRequirementCode.CATALOG,
            DoctorActionCode.SEED_DEMO,
        )
    repository = catalog.repository
    try:
        expected = materialize_synthetic_demo()
        dataset = repository.get_dataset_version(expected.dataset_version.id)
        query_set = repository.get_query_set_revision(expected.query_set.id)
        query_ids = repository.list_query_ids(query_set.id, limit=50)
        catalog_configs = repository.list_retrieval_configs(
            dataset_version_id=dataset.id,
            limit=5,
        )
        configs = [repository.get_retrieval_config(item.id) for item in expected.configs]
        run = repository.get_run(expected.completed_run.id)
        run_configs = repository.list_run_configs(run.id)
        outcomes = repository.list_outcomes(run.id, limit=200)
        expected_outcomes = sorted(
            expected.outcomes,
            key=lambda item: (str(item.query_id), str(item.config_id)),
        )
        if (
            dataset != expected.dataset_version
            or query_set != expected.query_set
            or query_ids != [item.judged_query.id for item in AUTHORED_SYNTHETIC_DEMO.queries]
            or len(catalog_configs) != 4
            or configs != list(expected.configs)
            or run != expected.completed_run
            or run_configs != list(expected.configs)
            or outcomes != expected_outcomes
        ):
            raise ValueError
    except Exception as error:
        _detach_exception(error)
        return _local_failure(
            DoctorCheckName.DEMO,
            DoctorRequirementCode.DEMO_EVIDENCE,
            DoctorActionCode.SEED_DEMO,
        )
    return DoctorCheck(
        name=DoctorCheckName.DEMO,
        state=DoctorCheckState.READY,
        dataset_version_id=dataset.id,
        dataset_count=1,
        query_count=query_set.query_count,
        config_count=len(configs),
        completed_run_count=1,
    )


def _inspect_live_tiny(
    settings: Settings,
    dependencies: DoctorDependencies,
) -> tuple[DoctorCheck, DoctorLiveTarget | None]:
    try:
        capability = dependencies.capability_inspector_factory(settings).inspect().live_playground
    except Exception as error:
        _detach_exception(error)
        return (
            _local_failure(
                DoctorCheckName.LIVE_TINY,
                DoctorRequirementCode.OWNED_TINY_RECEIPT,
                DoctorActionCode.RESOLVE_OWNED_TINY_RECEIPT,
            ),
            None,
        )
    if capability.state is CapabilityState.ACTION_REQUIRED:
        return (
            DoctorCheck(
                name=DoctorCheckName.LIVE_TINY,
                state=DoctorCheckState.ACTION_REQUIRED,
                requirements=capability.requirements,
                next_action=capability.next_action,
            ),
            None,
        )
    try:
        target = dependencies.owned_tiny_target_resolver(settings)
    except Exception as error:
        _detach_exception(error)
        target = None
    if target is None or not _valid_target(target):
        return (
            _local_failure(
                DoctorCheckName.LIVE_TINY,
                DoctorRequirementCode.OWNED_TINY_RECEIPT,
                DoctorActionCode.RESOLVE_OWNED_TINY_RECEIPT,
            ),
            None,
        )
    return (
        DoctorCheck(name=DoctorCheckName.LIVE_TINY, state=DoctorCheckState.READY),
        target,
    )


def _inspect_evaluation(
    catalog: ExistingReadOnlyCatalog | None,
    *,
    dataset_version_id: UUID | None,
    require_live: bool,
    local_region: str,
) -> tuple[DoctorCheck, DoctorLiveTarget | None]:
    if catalog is None:
        return (
            _local_failure(
                DoctorCheckName.EVALUATION,
                DoctorRequirementCode.CATALOG,
                DoctorActionCode.REPAIR_CATALOG,
            ),
            None,
        )
    repository = catalog.repository
    try:
        if dataset_version_id is None:
            datasets = repository.list_dataset_versions(limit=2)
            if len(datasets) != 1:
                return (
                    DoctorCheck(
                        name=DoctorCheckName.EVALUATION,
                        state=DoctorCheckState.ACTION_REQUIRED,
                        requirements=(DoctorRequirementCode.DATASET_SELECTION,),
                        next_action=DoctorActionCode.SELECT_DATASET,
                        dataset_count=len(datasets),
                    ),
                    None,
                )
            dataset = datasets[0]
            dataset_count = 1
        else:
            dataset = repository.get_dataset_version(dataset_version_id)
            dataset_count = 1
        return _validate_evaluation(
            repository,
            dataset,
            dataset_count=dataset_count,
            require_live=require_live,
            local_region=local_region,
        )
    except Exception as error:
        _detach_exception(error)
        return (
            _local_failure(
                DoctorCheckName.EVALUATION,
                DoctorRequirementCode.QUERY_SET,
                DoctorActionCode.REPAIR_CATALOG,
            ),
            None,
        )


def _validate_evaluation(
    repository: PufferLabRepository,
    dataset: DatasetVersion,
    *,
    dataset_count: int,
    require_live: bool,
    local_region: str,
) -> tuple[DoctorCheck, DoctorLiveTarget | None]:
    if dataset.status is not DatasetStatus.READY:
        return (
            _local_failure(
                DoctorCheckName.EVALUATION,
                DoctorRequirementCode.DATASET_READY,
                DoctorActionCode.COMPLETE_EVALUATION,
            ),
            None,
        )
    if require_live and dataset.data_origin is not DataOrigin.LIVE:
        return (
            _local_failure(
                DoctorCheckName.EVALUATION,
                DoctorRequirementCode.LIVE_DATASET,
                DoctorActionCode.SELECT_LIVE_DATASET,
            ),
            None,
        )

    query_sets = repository.list_query_sets(dataset_version_id=dataset.id, limit=2)
    if len(query_sets) != 1:
        return (
            _local_failure(
                DoctorCheckName.EVALUATION,
                DoctorRequirementCode.QUERY_SET,
                DoctorActionCode.REPAIR_CATALOG,
            ),
            None,
        )
    query_set = query_sets[0]
    query_ids = repository.list_query_ids(query_set.id, limit=50)
    catalog_configs = repository.list_retrieval_configs(dataset_version_id=dataset.id, limit=5)
    expected_configs = _canonical_configs(dataset)
    configs = (
        [repository.get_retrieval_config(item.id) for item in expected_configs]
        if expected_configs is not None
        else []
    )
    if (
        len(catalog_configs) != 4
        or configs != expected_configs
        or not _canonical_query_set(dataset, query_set, query_ids)
    ):
        return (
            _local_failure(
                DoctorCheckName.EVALUATION,
                DoctorRequirementCode.CONFIG_CATALOG,
                DoctorActionCode.REPAIR_CATALOG,
            ),
            None,
        )

    qualifying: list[EvalRun] = []
    views = EvaluationViewService(repository)
    expected_config_ids = [config.id for config in configs]
    expected_outcome_ids = {
        (config_id, query_id) for config_id in expected_config_ids for query_id in query_ids
    }
    for run in repository.list_runs(limit=100):
        if run.query_set.id != query_set.id or run.status is not EvalRunStatus.COMPLETED:
            continue
        if [run.baseline_config_id, *run.candidate_config_ids] != expected_config_ids:
            continue
        if repository.list_run_configs(run.id) != configs:
            continue
        try:
            view = views.get_eval_run(run.id).result
        except Exception as error:
            _detach_exception(error)
            continue
        outcomes = repository.list_outcomes(run.id, limit=200)
        if (
            view.run != run
            or view.dataset_version_id != dataset.id
            or [item.id for item in view.configs] != expected_config_ids
            or len(outcomes) != 200
            or {(item.config_id, item.query_id) for item in outcomes} != expected_outcome_ids
        ):
            continue
        qualifying.append(run)
    if not qualifying:
        return (
            _local_failure(
                DoctorCheckName.EVALUATION,
                DoctorRequirementCode.COMPLETED_RUN,
                DoctorActionCode.COMPLETE_EVALUATION,
            ),
            None,
        )

    target: DoctorLiveTarget | None = None
    if require_live:
        regions = {run.environment.turbopuffer_region for run in qualifying}
        namespace = dataset.namespace
        if len(regions) != 1:
            return (
                _local_failure(
                    DoctorCheckName.EVALUATION,
                    DoctorRequirementCode.REGION,
                    DoctorActionCode.CONFIGURE_REGION,
                ),
                None,
            )
        stored_region = next(iter(regions))
        if not _valid_namespace(namespace):
            return (
                _local_failure(
                    DoctorCheckName.EVALUATION,
                    DoctorRequirementCode.NAMESPACE,
                    DoctorActionCode.REPAIR_CATALOG,
                ),
                None,
            )
        if not is_valid_metadata_probe_region(stored_region):
            return (
                _local_failure(
                    DoctorCheckName.EVALUATION,
                    DoctorRequirementCode.REGION,
                    DoctorActionCode.REPAIR_CATALOG,
                ),
                None,
            )
        if local_region != stored_region:
            return (
                _local_failure(
                    DoctorCheckName.EVALUATION,
                    DoctorRequirementCode.REGION_MATCH,
                    DoctorActionCode.CONFIGURE_REGION,
                ),
                None,
            )
        target = DoctorLiveTarget(namespace=namespace, region=stored_region)

    return (
        DoctorCheck(
            name=DoctorCheckName.EVALUATION,
            state=DoctorCheckState.READY,
            dataset_version_id=dataset.id,
            dataset_count=dataset_count,
            query_count=query_set.query_count,
            config_count=len(configs),
            completed_run_count=len(qualifying),
        ),
        target,
    )


def _canonical_query_set(
    dataset: DatasetVersion,
    query_set: QuerySet,
    query_ids: list[UUID],
) -> bool:
    if dataset.data_origin is DataOrigin.SYNTHETIC_DEMO:
        expected = materialize_synthetic_demo()
        return (
            dataset == expected.dataset_version
            and query_set == expected.query_set
            and query_ids == [item.judged_query.id for item in AUTHORED_SYNTHETIC_DEMO.queries]
        )
    try:
        curated = load_curated_query_manifest(_CURATED_MANIFEST)
        expected_query_ids = [
            uuid5(
                PUFFERLAB_NAMESPACE_UUID,
                f"judged-query:{dataset.version}:{entry.query_id}",
            )
            for entry in curated.entries
        ]
        expected_query_set_id = uuid5(
            PUFFERLAB_NAMESPACE_UUID,
            f"query-set:{dataset.id}:{curated.query_set_content_sha256}",
        )
        return (
            query_set.dataset_version_id == dataset.id
            and query_set.id == expected_query_set_id
            and query_set.name == "CQADupStack Unix curated 50"
            and query_set.version == curated.selection_version
            and query_set.query_count == 50
            and query_set.content_hash == curated.query_set_content_sha256
            and query_set.created_at == UNIX_REVISION_CREATED_AT
            and query_ids == expected_query_ids
        )
    except Exception as error:
        _detach_exception(error)
        return False


def _canonical_configs(dataset: DatasetVersion) -> list[RetrievalConfig] | None:
    if dataset.data_origin is DataOrigin.SYNTHETIC_DEMO:
        expected = materialize_synthetic_demo()
        if dataset != expected.dataset_version:
            return None
        return list(expected.configs)
    try:
        from pufferlab.datasets.cqadupstack import load_unix_dataset_manifest

        manifest = load_unix_dataset_manifest(_UNIX_MANIFEST)
        return list(
            derive_bound_retrieval_configs(
                dataset,
                manifest,
                namespace=dataset.namespace,
            )
        )
    except Exception as error:
        _detach_exception(error)
        return None


def _local_failure(
    name: DoctorCheckName,
    requirement: DoctorRequirementCode,
    action: DoctorActionCode,
) -> DoctorCheck:
    return DoctorCheck(
        name=name,
        state=DoctorCheckState.ACTION_REQUIRED,
        requirements=(requirement,),
        next_action=action,
    )


def _valid_namespace(namespace: str) -> bool:
    if not namespace.strip():
        return False
    try:
        return len(namespace.encode("utf-8")) <= _MAX_NAMESPACE_BYTES
    except UnicodeEncodeError:
        return False


def _valid_target(target: DoctorLiveTarget) -> bool:
    return _valid_namespace(target.namespace) and is_valid_metadata_probe_region(target.region)


async def _probe_explicit_target(
    settings: Settings,
    *,
    target: DoctorLiveTarget,
    probe: MetadataProbe,
) -> _ExplicitProbeOutcome:
    secret = settings.turbopuffer_api_key
    api_key = ""
    result: MetadataProbeResult | None = None
    check: DoctorCheck | None = None
    control = _ProbeControl.NONE
    if secret is None:
        check = _local_failure(
            DoctorCheckName.METADATA,
            DoctorRequirementCode.API_KEY,
            DoctorActionCode.CONFIGURE_API_KEY,
        )
    else:
        try:
            api_key = secret.get_secret_value()
        except (KeyboardInterrupt, asyncio.CancelledError) as error:
            _detach_exception(error)
            control = _ProbeControl.CANCELLED
        except SystemExit as error:
            _detach_exception(error)
            control = _ProbeControl.INTERNAL
        except Exception as error:
            _detach_exception(error)
            check = _local_failure(
                DoctorCheckName.METADATA,
                DoctorRequirementCode.API_KEY,
                DoctorActionCode.CONFIGURE_API_KEY,
            )
        except BaseException as error:
            _detach_exception(error)
            control = _ProbeControl.INTERNAL
        else:
            try:
                result = await probe(
                    api_key=api_key,
                    region=target.region,
                    namespace=target.namespace,
                )
            except (KeyboardInterrupt, asyncio.CancelledError) as error:
                _detach_exception(error)
                control = _ProbeControl.CANCELLED
            except SystemExit as error:
                _detach_exception(error)
                control = _ProbeControl.INTERNAL
            except MetadataProbeConfigurationError as error:
                _detach_exception(error)
                check = _local_failure(
                    DoctorCheckName.METADATA,
                    DoctorRequirementCode.REGION,
                    DoctorActionCode.CONFIGURE_REGION,
                )
            except Exception as error:
                _detach_exception(error)
                result = MetadataProbeResult(state=MetadataProbeState.REMOTE_FAILURE)
            except BaseException as error:
                _detach_exception(error)
                control = _ProbeControl.INTERNAL

    if result is not None:
        up_to_date = result.state is MetadataProbeState.INDEX_UP_TO_DATE
        updating = result.state is MetadataProbeState.INDEX_UPDATING
        check = DoctorCheck(
            name=DoctorCheckName.METADATA,
            state=(DoctorCheckState.READY if up_to_date else DoctorCheckState.REMOTE_FAILURE),
            metadata_reachable=result.metadata_reachable,
            index_up_to_date=up_to_date,
            index_updating=updating,
        )

    # This is the sole raw-key bridge. Every caught outcome converges here before a report or
    # process-control result can leave the frame.
    api_key = ""
    secret = None
    settings = cast(Settings, None)
    target = cast(DoctorLiveTarget, None)
    probe = cast(MetadataProbe, None)
    result = None

    if control is not _ProbeControl.NONE:
        return _ExplicitProbeOutcome(control=control)
    if check is None:
        return _ExplicitProbeOutcome(
            check=_local_failure(
                DoctorCheckName.METADATA,
                DoctorRequirementCode.API_KEY,
                DoctorActionCode.CONFIGURE_API_KEY,
            )
        )
    return _ExplicitProbeOutcome(check=check)


def _detach_exception(error: BaseException) -> None:
    error.__traceback__ = None
    error.__context__ = None
    error.__cause__ = None
