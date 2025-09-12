from ert.plugins import ErtRuntimePlugins
from ert.plugins.plugin_manager import hook_specification


@hook_specification
def site_configurations() -> ErtRuntimePlugins:  # type: ignore
    """
    :return: dict of ert runtime plugins
    """
