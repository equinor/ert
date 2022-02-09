import logging

import pluggy

from ert3.config import ConfigPluginRegistry

_PLUGIN_NAMESPACE = "ert3"

hook_implementation = pluggy.HookimplMarker(_PLUGIN_NAMESPACE)
hook_specification = pluggy.HookspecMarker(_PLUGIN_NAMESPACE)

# Imports below hook_implementation and hook_specification to avoid circular imports
import ert3.plugins.hook_specifications
import ert3.config.plugins.implementations


class ErtPluginManager(pluggy.PluginManager):
    def __init__(self, plugins=None):
        super().__init__(_PLUGIN_NAMESPACE)
        self.add_hookspecs(ert3.plugins.hook_specifications)
        if plugins is None:
            self.register(ert3.config.plugins.implementations)
            self.load_setuptools_entrypoints(_PLUGIN_NAMESPACE)
        else:
            for plugin in plugins:
                self.register(plugin)
        logging.debug(str(self))

    def __str__(self):
        self_str = "ERT Plugin manager:\n"
        for plugin in self.get_plugins():
            self_str += "\t" + self.get_name(plugin) + "\n"
            for hook_caller in self.get_hookcallers(plugin):
                self_str += "\t\t" + str(hook_caller) + "\n"
        return self_str

    def get_plugin_configs(self, registry: ConfigPluginRegistry):
        result = self.hook.configs(registry=registry)
        return result
