from ert_shared.plugins.plugin_manager import hook_specification


@hook_specification
def site_config_lines():
    """
    :return: List of lines to append to site config file
    :rtype: PluginResponse with data as list[str]
    """
