from ert3.plugins.plugin_manager import hook_specification


@hook_specification
def legacy_ertscript_workflow(config):
    """
    This hook allows the user to register a workflow with the config object. A workflow
    must add the class inheriting from ErtScript and an optional name.

    :param config: A handle to the main workflow config.
    :type config: :class:`ert_shared.plugins.workflow_config.WorkflowConfigs`
    :return: None
    """
