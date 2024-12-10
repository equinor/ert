from __future__ import annotations

from typing import TYPE_CHECKING, no_type_check

from ert.plugins.plugin_manager import hook_specification

if TYPE_CHECKING:
    from ert.plugins.plugin_response import PluginResponse
    from ert.plugins.workflow_config import WorkflowConfigs


@no_type_check
@hook_specification
def installable_jobs() -> PluginResponse[dict[str, str]]:
    """
    :return: dict with job names as keys and path to config as value
    :rtype: PluginResponse with data as dict[str,str]
    """


@no_type_check
@hook_specification(firstresult=True)
def job_documentation(job_name: str) -> PluginResponse[dict[str, str] | None]:
    """
    :return: If job_name is from your plugin return
             dict with documentation fields as keys and corresponding
             text as value (See below for details), else None.

    Valid fields:
    description: RST markdown as a string. Example: "This is a **dummy** description"
    examples: RST markdown as a string. Example: "This is an example"
    category: Dot seperated list categories (main_category.sub_category) for job.
              Example: "simulator.reservoir". When generating documentation in ERT the
              main category (category before the first dot) will be used to group
              the jobs into sections.
    """


@no_type_check
@hook_specification
def installable_workflow_jobs() -> PluginResponse[dict[str, str]]:
    """
    :return: dict with workflow job names as keys and path to config as value
    """


@no_type_check
@hook_specification
def legacy_ertscript_workflow(config: WorkflowConfigs) -> None:
    """
    This hook allows the user to register a workflow with the config object. A workflow
    must add the class inheriting from ErtScript and an optional name.

    :param config: A handle to the main workflow config.
    """
