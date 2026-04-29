from enum import StrEnum


class HookRuntime(StrEnum):
    PRE_SIMULATION = "PRE_SIMULATION"
    POST_SIMULATION = "POST_SIMULATION"
    PRE_UPDATE = "PRE_UPDATE"
    POST_UPDATE = "POST_UPDATE"
    PRE_FIRST_UPDATE = "PRE_FIRST_UPDATE"
    PRE_EXPERIMENT = "PRE_EXPERIMENT"
    POST_EXPERIMENT = "POST_EXPERIMENT"

    def workflow_tab_title(self) -> str:
        return {
            HookRuntime.PRE_EXPERIMENT: "Pre-experiment workflows",
            HookRuntime.POST_EXPERIMENT: "Post-experiment workflows",
            HookRuntime.PRE_SIMULATION: "Pre-simulation workflows",
            HookRuntime.POST_SIMULATION: "Post-simulation workflows",
            HookRuntime.PRE_FIRST_UPDATE: "Pre-first-update workflows",
            HookRuntime.PRE_UPDATE: "Pre-update workflows",
            HookRuntime.POST_UPDATE: "Post-update workflows",
        }[self]
