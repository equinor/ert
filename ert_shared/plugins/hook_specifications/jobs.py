from ert_shared.plugins.plugin_manager import hook_specification


@hook_specification
def installable_jobs():
    """
    :return: dict with job names as keys and path to config as value
    :rtype: PluginResponse with data as dict[str,str]
    """


@hook_specification
def installable_workflow_jobs():
    """
    :return: dict with workflow job names as keys and path to config as value
    :rtype: PluginResponse with data as dict[str,str]
    """

