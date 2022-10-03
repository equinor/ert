from typing import TYPE_CHECKING, Iterator, List

from .plugin import Plugin

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain
    from ert._c_wrappers.job_queue import WorkflowJob


class PluginHandler:
    def __init__(
        self, ert: "EnKFMain", plugin_jobs: List["WorkflowJob"], parent_window
    ):
        self.__ert = ert
        self.__plugins = []

        for job in plugin_jobs:
            plugin = Plugin(self.__ert, job)
            self.__plugins.append(plugin)
            plugin.setParentWindow(parent_window)

        self.__plugins = sorted(self.__plugins, key=Plugin.getName)

    def ert(self) -> "EnKFMain":
        return self.__ert

    def __iter__(self) -> Iterator[Plugin]:
        index = 0
        while index < len(self.__plugins):
            yield self.__plugins[index]
            index += 1

    def __getitem__(self, index) -> Plugin:
        return self.__plugins[index]

    def __len__(self):
        return len(self.__plugins)
