from __future__ import annotations

import functools
import typing
from typing import TYPE_CHECKING, NotRequired, TypedDict

from PyQt6.QtWidgets import QWidget

from ert.config.parsing.hook_runtime import HookRuntime

if TYPE_CHECKING:
    from ert.config import ESSettings, UpdateSettings
    from ert.runpaths import Runpaths
    from ert.storage import Ensemble, Storage


class PreExperimentFixtures(TypedDict):
    random_seed: int
    hook: HookRuntime = HookRuntime.PRE_EXPERIMENT


class PostExperimentFixtures(TypedDict):
    hook: HookRuntime = HookRuntime.POST_EXPERIMENT
    random_seed: int
    storage: Storage
    ensemble: Ensemble


class PreSimulationFixtures(TypedDict):
    hook: HookRuntime = HookRuntime.PRE_SIMULATION
    random_seed: int
    reports_dir: str
    run_paths: Runpaths
    storage: Storage
    ensemble: Ensemble


class PostSimulationFixtures(PreSimulationFixtures):
    hook: HookRuntime = HookRuntime.POST_SIMULATION


class PreFirstUpdateFixtures(TypedDict):
    hook: HookRuntime = HookRuntime.PRE_FIRST_UPDATE
    random_seed: int
    reports_dir: str
    run_paths: Runpaths
    storage: Storage
    ensemble: Ensemble
    es_settings: ESSettings
    observation_settings: UpdateSettings


class PreUpdateFixtures(PreFirstUpdateFixtures):
    hook: HookRuntime = HookRuntime.PRE_UPDATE


class PostUpdateFixtures(PreFirstUpdateFixtures):
    hook: HookRuntime = HookRuntime.POST_UPDATE


class WorkflowFixtures(TypedDict, total=False):
    workflow_args: NotRequired[list[typing.Any]]
    parent: NotRequired[QWidget]
    random_seed: NotRequired[int]
    reports_dir: NotRequired[str]
    run_paths: NotRequired[Runpaths]
    storage: NotRequired[Storage]
    ensemble: NotRequired[Ensemble]
    es_settings: NotRequired[ESSettings]
    observation_settings: NotRequired[UpdateSettings]


# Union Type Definition
HookedWorkflowFixtures = (
    PreExperimentFixtures
    | PostExperimentFixtures
    | PreSimulationFixtures
    | PostSimulationFixtures
    | PreFirstUpdateFixtures
    | PreUpdateFixtures
    | PostUpdateFixtures
)


def __all_workflow_fixtures() -> set[str]:
    fixtures_per_runtime = (
        __get_available_fixtures_for_hook(hook) for hook in HookRuntime
    )

    return functools.reduce(lambda a, b: a | b, fixtures_per_runtime)


def __get_available_fixtures_for_hook(hook: HookRuntime) -> set[str]:
    cls = next(
        cls for cls in typing.get_args(HookedWorkflowFixtures) if cls.hook == hook
    )

    return set(cls.__annotations__.keys()) - {"hook"}


all_hooked_workflow_fixtures = __all_workflow_fixtures()
fixtures_per_hook = {
    hook: __get_available_fixtures_for_hook(hook) for hook in HookRuntime
}
