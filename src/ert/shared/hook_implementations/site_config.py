from ert.shared.plugins.plugin_manager import hook_implementation
from ert.shared.plugins.plugin_response import plugin_response


@hook_implementation
@plugin_response(plugin_name="ert")
def site_config_lines():
    return [
        "JOB_SCRIPT job_dispatch.py",
        "QUEUE_SYSTEM LOCAL",
        "QUEUE_OPTION LOCAL MAX_RUNNING 1",
    ]
