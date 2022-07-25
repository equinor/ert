from ert_shared.plugins.plugin_manager import hook_specification


@hook_specification(firstresult=True)
def ecl100_config_path():
    """
    :return: Path to ecl100 config file
    :rtype: PluginResponse with data as str
    """


@hook_specification(firstresult=True)
def ecl300_config_path():
    """
    :return: Path to ecl300 config file
    :rtype: PluginResponse with data as str
    """


@hook_specification(firstresult=True)
def flow_config_path():
    """
    :return: Path to flow config file
    :rtype: PluginResponse with data as str
    """


@hook_specification(firstresult=True)
def rms_config_path():
    """
    :return: Path to flow config file
    :rtype: PluginResponse with data as str
    """
