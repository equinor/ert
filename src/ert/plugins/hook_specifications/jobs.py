from __future__ import annotations

from typing import TYPE_CHECKING, no_type_check

from ert.plugins.plugin_manager import hook_specification

if TYPE_CHECKING:
    from ert.config import WorkflowConfigs
    from ert.plugins.plugin_response import PluginResponse


@no_type_check
@hook_specification
def installable_workflow_jobs() -> PluginResponse[dict[str, str]]:
    """:return: dict with workflow job names as keys and path to config as value"""


@no_type_check
@hook_specification
def ertscript_workflow(config: WorkflowConfigs) -> None:
    """
    This hook allows the user to register a workflow with the config object. A workflow
    must add the class inheriting from ErtScript and an optional name.

    :param config: A handle to the main workflow config.
    """


@no_type_check
@hook_specification
def legacy_ertscript_workflow(config: WorkflowConfigs) -> None:
    """Deprecated Variant of the hook ertscript_workflow"""
