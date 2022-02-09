from ert3.plugins.plugin_manager import hook_specification
from ert3.config import ConfigPluginRegistry

@hook_specification
def configs(registry: ConfigPluginRegistry) -> None:
    """
    This hook allows the user to register plugin configs with the config registry.
    A Plugin Config must add the class inheriting from ErtScript and an optional name.
    """
