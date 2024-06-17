from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QMenu

from ert.gui.tools import Tool

from .plugin_runner import PluginRunner

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.gui.ertnotifier import ErtNotifier

    from .plugin_handler import PluginHandler


class PluginsTool(Tool):
    def __init__(
        self,
        plugin_handler: PluginHandler,
        notifier: ErtNotifier,
        ert_config: ErtConfig,
    ) -> None:
        enabled = len(plugin_handler) > 0
        self.notifier = notifier
        super().__init__(
            "Plugins",
            QIcon("img:widgets.svg"),
            enabled,
            popup_menu=True,
        )

        self.__plugins = {}

        menu = QMenu()
        for plugin in plugin_handler:
            plugin_runner = PluginRunner(plugin, ert_config, notifier.storage)
            plugin_runner.setPluginFinishedCallback(self.trigger)

            self.__plugins[plugin] = plugin_runner
            plugin_action = menu.addAction(plugin.getName())
            assert plugin_action is not None
            plugin_action.setToolTip(plugin.getDescription())
            plugin_action.triggered.connect(plugin_runner.run)

        self.getAction().setMenu(menu)

    def trigger(self) -> None:
        self.notifier.emitErtChange()  # plugin may have added new cases.

    def get_plugin_runner(self, plugin_name: str) -> Optional[PluginRunner]:
        for pulgin, runner in self.__plugins.items():
            if pulgin.getName() == plugin_name:
                return runner
        return None
