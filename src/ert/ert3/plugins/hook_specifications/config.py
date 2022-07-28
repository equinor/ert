from ert.ert3.plugins.plugin_manager import hook_specification
from ert.ert3.config import ConfigPluginRegistry


@hook_specification  # type: ignore
def configs(registry: ConfigPluginRegistry) -> None:
    """
    This hook allows the user to register plugin configs with the config registry.
    Plugins belong to a category and are identified using a name. See
    :class:`ConfigPluginRegistry` for more details.
    """
