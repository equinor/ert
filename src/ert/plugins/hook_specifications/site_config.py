from typing import List

from ert.plugins.plugin_manager import hook_specification


@hook_specification
def site_config_lines() -> List[str]:  # type: ignore
    """
    :return: List of lines to append to site config file
    """
