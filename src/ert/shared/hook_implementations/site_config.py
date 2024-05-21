from __future__ import annotations
import os
from ert.shared.plugins.plugin_manager import hook_implementation
from ert.shared.plugins.plugin_response import plugin_response


@hook_implementation
@plugin_response(plugin_name="ert")  # type: ignore
def site_config_lines():
    return [
        "JOB_SCRIPT job_dispatch.py",
        "QUEUE_SYSTEM LOCAL",
        "QUEUE_OPTION LOCAL MAX_RUNNING 1",
        "FORWARD_MODEL azure_instance",
        f"LOAD_WORKFLOW {os.path.dirname(__file__)}/workflows/AZURE_COST",
        "HOOK_WORKFLOW AZURE_COST POST_SIMULATION",
    ]
