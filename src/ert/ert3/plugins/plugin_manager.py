from typing import Any, List, Optional

import pluggy
from ert.ert3.config import ConfigPluginRegistry

_PLUGIN_NAMESPACE = "ert3"

hook_implementation = pluggy.HookimplMarker(_PLUGIN_NAMESPACE)
hook_specification = pluggy.HookspecMarker(_PLUGIN_NAMESPACE)


# Imports below hook_implementation and hook_specification to avoid circular imports
# pylint: disable=C0413
from ert.ert3.config.plugins import implementations  # noqa: E402
from ert.ert3.plugins import hook_specifications  # noqa: E402


# type ignored due to pluggy lacking type information
class ErtPluginManager(pluggy.PluginManager):  # type: ignore
    def __init__(self, plugins: Optional[List[Any]] = None) -> None:
        super().__init__(_PLUGIN_NAMESPACE)
        self.add_hookspecs(hook_specifications)
        if plugins is None:
            self.register(implementations)
            self.load_setuptools_entrypoints(_PLUGIN_NAMESPACE)
        else:
            for plugin in plugins:
                self.register(plugin)

    def collect(self, registry: ConfigPluginRegistry) -> None:
        self.hook.configs(registry=registry)  # pylint: disable=E1101
