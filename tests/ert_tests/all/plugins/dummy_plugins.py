from ert.shared.plugins.plugin_manager import hook_implementation
from ert.shared.plugins.plugin_response import plugin_response


@hook_implementation
@plugin_response(plugin_name="dummy")
def help_links():
    return {"test": "test", "test2": "test"}


@hook_implementation
@plugin_response(plugin_name="dummy")
def ecl100_config_path():
    return "/dummy/path/ecl100_config.yml"


@hook_implementation
@plugin_response(plugin_name="dummy")
def ecl300_config_path():
    return "/dummy/path/ecl300_config.yml"


@hook_implementation
@plugin_response(plugin_name="dummy")
def flow_config_path():
    return "/dummy/path/flow_config.yml"


@hook_implementation
@plugin_response(plugin_name="dummy")
def rms_config_path():
    return "/dummy/path/rms_config.yml"


@hook_implementation
@plugin_response(plugin_name="dummy")
def installable_jobs():
    return {"job1": "/dummy/path/job1", "job2": "/dummy/path/job2"}


@hook_implementation
@plugin_response(plugin_name="dummy")
def installable_workflow_jobs():
    return {"wf_job1": "/dummy/path/wf_job1", "wf_job2": "/dummy/path/wf_job2"}


@hook_implementation
@plugin_response(plugin_name="dummy")
def legacy_ertscript_workflow(config):
    def some_func():
        pass

    config.add_workflow(some_func)


@hook_implementation
@plugin_response(plugin_name="dummy")
def site_config_lines():
    return ["QUEUE_OPTION LOCAL MAX_RUNNING 2"]


@hook_implementation
@plugin_response(plugin_name="dummy")
def job_documentation(job_name):
    if job_name == "job1":
        return {
            "description": "job description",
            "examples": "example 1 and example 2",
            "category": "test.category.for.job",
        }
    return None


class ExamplePlugin:
    name = "example"

    @staticmethod
    def run():
        pass


@hook_implementation
def add_log_handle_to_root():
    import logging

    fh = logging.FileHandler("spam.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    return fh
