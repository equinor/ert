from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from ert.config import ESSettings, UpdateSettings
from ert.runpaths import Runpaths
from ert.storage import Ensemble, Storage


class HookRuntime(StrEnum):
    PRE_SIMULATION = "PRE_SIMULATION"
    POST_SIMULATION = "POST_SIMULATION"
    PRE_UPDATE = "PRE_UPDATE"
    POST_UPDATE = "POST_UPDATE"
    PRE_FIRST_UPDATE = "PRE_FIRST_UPDATE"
    PRE_EXPERIMENT = "PRE_EXPERIMENT"
    POST_EXPERIMENT = "POST_EXPERIMENT"

@dataclass
class PreExperimentFixtures:
    random_seed: int
    hook: Literal["pre_experiment"] = HookRuntime.PRE_EXPERIMENT


class PostExperimentFixtures(PreExperimentFixtures):
    storage: Storage
    ensemble: Ensemble
    hook: HookRuntime = HookRuntime.POST_EXPERIMENT

class PreSimulationFixtures(PostExperimentFixtures):
    reports_dir: str
    run_paths: Runpaths
    hook: HookRuntime = HookRuntime.PRE_SIMULATION

class PostSimulationFixtures(PreSimulationFixtures):
    hook: HookRuntime = HookRuntime.POST_SIMULATION


class PreFirstUpdateFixtures(PreSimulationFixtures):
    es_settings: ESSettings
    observation_settings: UpdateSettings
    hook: HookRuntime = HookRuntime.PRE_FIRST_UPDATE


class PreUpdateFixtures(PreFirstUpdateFixtures):
    hook: HookRuntime = HookRuntime.PRE_UPDATE


class PostUpdateFixtures(PreFirstUpdateFixtures):
    hook: HookRuntime = HookRuntime.POST_UPDATE


WorkflowFixtures = PreExperimentFixtures | PostExperimentFixtures | PreSimulationFixtures | PostSimulationFixtures | PreFirstUpdateFixtures | PreUpdateFixtures | PostUpdateFixtures

fixtures_per_runtime = {
    HookRuntime.PRE_EXPERIMENT: PreExperimentFixtures.__,
    HookRuntime.PRE_SIMULATION: {
        "storage",
        "ensemble",
        "reports_dir",
        "random_seed",
        "run_paths",
    },
    HookRuntime.POST_SIMULATION: {
        "storage",
        "ensemble",
        "reports_dir",
        "random_seed",
        "run_paths",
    },
    HookRuntime.PRE_FIRST_UPDATE: {
        "storage",
        "ensemble",
        "reports_dir",
        "random_seed",
        "es_settings",
        "observation_settings",
        "run_paths",
    },
    HookRuntime.PRE_UPDATE: {
        "storage",
        "ensemble",
        "reports_dir",
        "random_seed",
        "es_settings",
        "observation_settings",
        "run_paths",
    },
    HookRuntime.POST_UPDATE: {
        "storage",
        "ensemble",
        "reports_dir",
        "random_seed",
        "es_settings",
        "observation_settings",
        "run_paths",
    },
    HookRuntime.POST_EXPERIMENT: {
        "random_seed",
        "storage",
        "ensemble",
    },
}
