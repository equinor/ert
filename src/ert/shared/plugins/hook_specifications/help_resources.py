from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from ert.shared.plugins.plugin_manager import hook_specification

if TYPE_CHECKING:
    from ert.shared.plugins.plugin_response import PluginResponse


@hook_specification
def help_links() -> PluginResponse[Dict[str, str]]:  # type: ignore
    """Have a look at the ingredients and offer your own.

    :return: Dictionary with link as values and link labels as keys
    """
