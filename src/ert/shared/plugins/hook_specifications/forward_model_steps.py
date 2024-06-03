from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Type, no_type_check

from ert.config import ForwardModelStepPlugin
from ert.shared.plugins.plugin_manager import hook_specification

if TYPE_CHECKING:
    from ert.shared.plugins.plugin_response import PluginResponse


@no_type_check
@hook_specification
def installable_forward_model_steps() -> (
    PluginResponse[List[Type[ForwardModelStepPlugin]]]
):
    """
    :return: List of forward model step plugins in the form of subclasses of the
        ForwardModelStepPlugin class
    """


@no_type_check
@hook_specification(firstresult=True)
def forward_model_step_documentation(
    forward_model_step_name: str, description: str, examples: str, category: str
) -> PluginResponse[Optional[Dict[str, str]]]:
    """
    Valid fields:

    description:
        RST markdown as a string.
        Example: "This is a **dummy** description"

    examples:
        RST markdown as a string.
        Example: "This is an example"

    category:
        Dot separated list categories (main_category.sub_category) for forward model.
        Example: "simulator.reservoir"
        When generating documentation in ERT, the main category
        (category before the first dot) will be used to group the forward
        models into sections.

    Returns:
        dict or None: If `forward_model_step_name` is from your plugin, returns a
        dictionary with documentation fields as keys and corresponding text as values,
        else None.
    """
