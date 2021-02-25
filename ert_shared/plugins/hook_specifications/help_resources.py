from ert_shared.plugins.plugin_manager import hook_specification


@hook_specification
def help_links():
    """Have a look at the ingredients and offer your own.

    :return: Dictionary with link as values and link labels as keys
    :rtype: PluginResponse with data as dict[str,str]
    """
