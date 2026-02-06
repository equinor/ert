from ert.plugins.plugin_manager import hook_specification


@hook_specification
def get_ip_address() -> str:  # type: ignore
    pass
