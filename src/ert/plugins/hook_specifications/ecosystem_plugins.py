from __future__ import annotations

from typing import TYPE_CHECKING, no_type_check

from ert.plugins.plugin_manager import hook_specification

if TYPE_CHECKING:
    from ert.config import EcosystemPlugin
    from ert.plugins.plugin_response import PluginResponse


@no_type_check
@hook_specification
def ecosystem_plugin() -> PluginResponse[type[EcosystemPlugin]]:
    """
    :return: The ecosystem plugin in the form of a subclass of the
        EcosystemPlugin class. It is selectable from the user config with
        ``ECOSYSTEM <name>``. Installing more than one is an error.
    """
