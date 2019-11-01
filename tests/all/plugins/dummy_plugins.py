from ert_shared.plugins.plugin_manager import hook_implementation
from ert_shared.plugins.plugin_response import plugin_response


@hook_implementation
@plugin_response(plugin_name="ert")
def help_links():
    return {"test": "test", "test2": "test"}


@hook_implementation
@plugin_response(plugin_name="ert")
def ecl100_config_path():
    return "/dummy/path/ecl100_config.yml"


@hook_implementation
@plugin_response(plugin_name="ert")
def ecl300_config_path():
    return "/dummy/path/ecl300_config.yml"


@hook_implementation
@plugin_response(plugin_name="ert")
def flow_config_path():
    return "/dummy/path/flow_config.yml"


@hook_implementation
@plugin_response(plugin_name="ert")
def rms_config_path():
    return "/dummy/path/rms_config.yml"


@hook_implementation
@plugin_response(plugin_name="ert")
def installable_jobs():
    return {"job1": "/dummy/path/job1", "job2": "/dummy/path/job2"}


@hook_implementation
@plugin_response(plugin_name="ert")
def installable_workflow_jobs():
    return {"wf_job1": "/dummy/path/wf_job1", "wf_job2": "/dummy/path/wf_job2"}

@hook_implementation
@plugin_response(plugin_name="ert")
def site_config_lines():
    return [
        "JOB_SCRIPT job_dispatch_dummy.py",
        "QUEUE_OPTION LOCAL MAX_RUNNING 2",
    ]

