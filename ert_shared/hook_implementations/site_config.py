from os import uname

from ert_shared.plugins.plugin_manager import hook_implementation
from ert_shared.plugins.plugin_response import plugin_response


def _lib_extension():
    return "dylib" if uname()[0] == "Darwin" else "so"


@hook_implementation
@plugin_response(plugin_name="ert")
def site_config_lines():
    return [
        "JOB_SCRIPT job_dispatch.py",
        "QUEUE_SYSTEM LOCAL",
        "QUEUE_OPTION LOCAL MAX_RUNNING 1",
        "ANALYSIS_LOAD RML_ENKF rml_enkf.{}".format(_lib_extension()),
    ]
