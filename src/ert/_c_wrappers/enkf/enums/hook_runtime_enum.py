from cwrap import BaseCEnum


class HookRuntime(BaseCEnum):
    TYPE_NAME = "hook_runtime_enum"
    PRE_SIMULATION = 0
    POST_SIMULATION = 1
    PRE_UPDATE = 2
    POST_UPDATE = 3
    PRE_FIRST_UPDATE = 4


HookRuntime.addEnum("PRE_SIMULATION", 0)
HookRuntime.addEnum("POST_SIMULATION", 1)
HookRuntime.addEnum("PRE_UPDATE", 2)
HookRuntime.addEnum("POST_UPDATE", 3)
HookRuntime.addEnum("PRE_FIRST_UPDATE", 4)
