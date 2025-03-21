from __future__ import annotations

import typing
from typing import TYPE_CHECKING, TypedDict
from ert.config.parsing.hook_runtime import HookRuntime

if TYPE_CHECKING:
    from ert.config import ESSettings, UpdateSettings
    from ert.runpaths import Runpaths
    from ert.storage import Ensemble, Storage


class _BaseWorkflowFixtures(TypedDict):
    random_seed: int


class _UpdateWorkflowFixtures(TypedDict):
    es_settings: ESSettings
    observation_settings: UpdateSettings


class _StorageWorkflowFixtures(TypedDict):
    storage: Storage
    ensemble: Ensemble


class PreExperimentFixtures(_BaseWorkflowFixtures):
    random_seed: int
    hook: HookRuntime.PRE_EXPERIMENT


class PostExperimentFixtures(_BaseWorkflowFixtures, _StorageWorkflowFixtures):
    hook: HookRuntime.POST_EXPERIMENT


class PreSimulationFixtures(_BaseWorkflowFixtures, _StorageWorkflowFixtures):
    hook: HookRuntime.PRE_SIMULATION
    reports_dir: str
    run_paths: Runpaths


class PostSimulationFixtures(PreSimulationFixtures):
    hook: HookRuntime.POST_SIMULATION


class PreFirstUpdateFixtures(
    _BaseWorkflowFixtures, _UpdateWorkflowFixtures, _StorageWorkflowFixtures
):
    hook: HookRuntime.PRE_FIRST_UPDATE
    reports_dir: str
    run_paths: Runpaths


class PreUpdateFixtures(PreFirstUpdateFixtures):
    hook: HookRuntime.PRE_UPDATE


class PostUpdateFixtures(PreFirstUpdateFixtures):
    hook: HookRuntime.POST_UPDATE


# Union Type Definition
WorkflowFixtures = (
    PreExperimentFixtures
    | PostExperimentFixtures
    | PreSimulationFixtures
    | PostSimulationFixtures
    | PreFirstUpdateFixtures
    | PreUpdateFixtures
    | PostUpdateFixtures
)


def matches_hook(cls, hook: HookRuntime):
    return cls.__annotations__.get("hook") == hook


def available_fixtures(cls):
    return set(cls.__annotations__.keys()) - {"hook"}


def get_available_fixtures_for_runtime(hook: HookRuntime) -> set[str]:
    cls = next(
        cls
        for cls in typing.get_args(WorkflowFixtures)
        if matches_hook(cls, hook)
    )

    ref = cls

    return available_fixtures(cls)
