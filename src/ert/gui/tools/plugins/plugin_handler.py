from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from ert.config.workflow_job import ErtScriptWorkflow

from .plugin import Plugin

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

    from ert.gui.ertnotifier import ErtNotifier


class PluginHandler:
    def __init__(
        self,
        notifier: ErtNotifier,
        plugin_jobs: list[ErtScriptWorkflow],
        parent_window: QWidget | None,
    ) -> None:
        self.__plugins = []

        for job in plugin_jobs:
            plugin = Plugin(notifier, job)
            self.__plugins.append(plugin)
            plugin.setParentWindow(parent_window)

        self.__plugins = sorted(self.__plugins, key=Plugin.getName)

    def ert(self) -> None:
        raise NotImplementedError("No such property")

    def __iter__(self) -> Iterator[Plugin]:
        index = 0
        while index < len(self.__plugins):
            yield self.__plugins[index]
            index += 1

    def __getitem__(self, index: int) -> Plugin:
        return self.__plugins[index]

    def __len__(self) -> int:
        return len(self.__plugins)
