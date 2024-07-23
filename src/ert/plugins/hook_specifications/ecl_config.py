from __future__ import annotations

from typing import TYPE_CHECKING

from ert.plugins.plugin_manager import hook_specification

if TYPE_CHECKING:
    from ert.plugins.plugin_response import PluginResponse


@hook_specification(firstresult=True)
def ecl100_config_path() -> PluginResponse[str]:  # type: ignore
    """
    :return: Path to ecl100 config file
    """


@hook_specification(firstresult=True)
def ecl300_config_path() -> PluginResponse[str]:  # type: ignore
    """
    :return: Path to ecl300 config file
    """


@hook_specification(firstresult=True)
def flow_config_path() -> PluginResponse[str]:  # type: ignore
    """
    :return: Path to flow config file
    """
