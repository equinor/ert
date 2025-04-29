from __future__ import annotations

import functools
import operator
import typing
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal

from PyQt6.QtWidgets import QWidget
from typing_extensions import TypedDict

from ert.config.parsing.hook_runtime import HookRuntime

if TYPE_CHECKING:
    from ert.config import ESSettings, ObservationSettings
    from ert.runpaths import Runpaths
    from ert.storage import Ensemble, Storage


@dataclass
class PreExperimentFixtures:
    random_seed: int
    hook: Literal[HookRuntime.PRE_EXPERIMENT] = HookRuntime.PRE_EXPERIMENT


@dataclass
class PostExperimentFixtures:
    random_seed: int
    storage: Storage
    ensemble: Ensemble
    hook: Literal[HookRuntime.POST_EXPERIMENT] = HookRuntime.POST_EXPERIMENT


@dataclass
class PreSimulationFixtures:
    random_seed: int
    reports_dir: str
    run_paths: Runpaths
    storage: Storage
    ensemble: Ensemble
    hook: HookRuntime = HookRuntime.PRE_SIMULATION


@dataclass
class PostSimulationFixtures(PreSimulationFixtures):
    hook: Literal[HookRuntime.POST_SIMULATION] = HookRuntime.POST_SIMULATION


@dataclass
class PreFirstUpdateFixtures:
    random_seed: int
    reports_dir: str
    run_paths: Runpaths
    storage: Storage
    ensemble: Ensemble
    es_settings: ESSettings
    observation_settings: ObservationSettings
    hook: HookRuntime = HookRuntime.PRE_FIRST_UPDATE


@dataclass
class PreUpdateFixtures(PreFirstUpdateFixtures):
    hook: HookRuntime = HookRuntime.PRE_UPDATE


@dataclass
class PostUpdateFixtures(PreFirstUpdateFixtures):
    hook: HookRuntime = HookRuntime.POST_UPDATE


class WorkflowFixtures(TypedDict, total=False):
    workflow_args: list[typing.Any]
    parent: QWidget | None
    random_seed: int | None
    reports_dir: str
    run_paths: Runpaths
    storage: Storage
    ensemble: Ensemble | None
    es_settings: ESSettings
    observation_settings: ObservationSettings


def create_workflow_fixtures_from_hooked(
    hooked_fixtures: HookedWorkflowFixtures,
) -> WorkflowFixtures:
    fixtures = {}
    for k in WorkflowFixtures.__annotations__:
        if getattr(hooked_fixtures, k, None) is not None:
            fixtures[k] = getattr(hooked_fixtures, k)

    return typing.cast(WorkflowFixtures, fixtures)


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

    return functools.reduce(operator.or_, fixtures_per_runtime)


def __get_available_fixtures_for_hook(hook: HookRuntime) -> set[str]:
    cls = next(
        cls for cls in typing.get_args(HookedWorkflowFixtures) if cls.hook == hook
    )

    return {f.name for f in fields(cls)} - {"hook"}


all_hooked_workflow_fixtures = __all_workflow_fixtures()
fixtures_per_hook = {
    hook: __get_available_fixtures_for_hook(hook) for hook in HookRuntime
}
