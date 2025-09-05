from typing import Unpack

from ert.plugins import ErtRuntimePlugins
from ert.plugins.plugin_manager import hook_specification


@hook_specification
def site_config_lines() -> list[str]:  # type: ignore
    """
    :return: List of lines to append to site config file
    """


@hook_specification
def site_configurations() -> Unpack[ErtRuntimePlugins]:
    """
    :return: dict of ert runtime plugins
    """
