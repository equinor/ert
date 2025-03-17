from enum import StrEnum


class HookRuntime(StrEnum):
    PRE_SIMULATION = "PRE_SIMULATION"
    POST_SIMULATION = "POST_SIMULATION"
    PRE_UPDATE = "PRE_UPDATE"
    POST_UPDATE = "POST_UPDATE"
    PRE_FIRST_UPDATE = "PRE_FIRST_UPDATE"
    PRE_EXPERIMENT = "PRE_EXPERIMENT"
    POST_EXPERIMENT = "POST_EXPERIMENT"


fixtures_per_runtime = {
    HookRuntime.PRE_EXPERIMENT: {"random_seed"},
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
