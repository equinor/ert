from qtpy.QtWidgets import QMenu

from ert.gui.ertwidgets import resourceIcon
from ert.gui.tools import Tool

from .plugin_runner import PluginRunner


class PluginsTool(Tool):
    def __init__(self, plugin_handler, notifier):
        """
        @type plugin_handler: PluginHandler
        """
        enabled = len(plugin_handler) > 0
        self.notifier = notifier
        super().__init__(
            "Plugins",
            "tools/plugins",
            resourceIcon("widgets.svg"),
            enabled,
            popup_menu=True,
        )

        self.__plugins = {}

        menu = QMenu()
        for plugin in plugin_handler:
            plugin_runner = PluginRunner(plugin)
            plugin_runner.setPluginFinishedCallback(self.trigger)

            self.__plugins[plugin] = plugin_runner
            plugin_action = menu.addAction(plugin.getName())
            plugin_action.setToolTip(plugin.getDescription())
            plugin_action.triggered.connect(plugin_runner.run)

        self.getAction().setMenu(menu)

    def trigger(self):
        self.notifier.emitErtChange()  # plugin may have added new cases.
