from ert.plugins import ErtRuntimePlugins
from ert.plugins.plugin_manager import hook_specification, single_implementation


@single_implementation
@hook_specification
def site_configurations() -> ErtRuntimePlugins:  # type: ignore
    """Configure global ERT settings for a specific computing environment (a "site").

    Only one implementation is allowed.
    """
