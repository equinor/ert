from __future__ import annotations

from typing import TYPE_CHECKING, Any, no_type_check

from ert.plugins.plugin_manager import hook_specification

if TYPE_CHECKING:
    from ert.config import ForwardModelStepPlugin
    from ert.plugins.plugin_response import PluginResponse


@no_type_check
@hook_specification
def installable_forward_model_steps() -> PluginResponse[
    list[type[ForwardModelStepPlugin]]
]:
    """
    :return: List of forward model step plugins in the form of subclasses of the
        ForwardModelStepPlugin class
    """


@no_type_check
@hook_specification
def forward_model_configuration() -> PluginResponse[dict[str, dict[str, Any]]]:
    """Used to declare a forward model plugins configuration.

    Example:

       import ert

       @ert.plugin(name="my_plugin")
       def forward_model_configuration():
            return {
                "forward_model_step_name": {
                    "config_key" : <config value>
                },
            }
    """
