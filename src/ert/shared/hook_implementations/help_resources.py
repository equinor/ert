from ert.shared.plugins.plugin_manager import hook_implementation
from ert.shared.plugins.plugin_response import plugin_response


@hook_implementation
@plugin_response(plugin_name="ert")  # type: ignore
def help_links():
    return {"GitHub page": "https://github.com/equinor/ert"}
