import logging

from ert import ForwardModelStepPlugin, plugin


@plugin(name="dummy")
def help_links():
    return {"test": "test", "test2": "test"}


@plugin(name="dummy")
def forward_model_paths():
    return ["/foo/bin", "/bar/bin"]


@plugin(name="dummy")
def ecl100_config_path():
    return "/dummy/path/ecl100_config.yml"


@plugin(name="dummy")
def ecl300_config_path():
    return "/dummy/path/ecl300_config.yml"


@plugin(name="dummy")
def flow_config_path():
    return "/dummy/path/flow_config.yml"


@plugin(name="dummy")
def installable_jobs():
    return {"job1": "/dummy/path/job1", "job2": "/dummy/path/job2"}


@plugin(name="dummy")
def installable_workflow_jobs():
    return {"wf_job1": "/dummy/path/wf_job1", "wf_job2": "/dummy/path/wf_job2"}


@plugin(name="dummy")
def legacy_ertscript_workflow(config):
    def some_func():
        pass

    config.add_workflow(some_func)


@plugin(name="dummy")
def site_config_lines():
    return ["JOB_SCRIPT job_dispatch_dummy.py", "QUEUE_OPTION LOCAL MAX_RUNNING 2"]


@plugin(name="dummy")
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


@plugin(name="dummy")
def add_log_handle_to_root():
    fh = logging.FileHandler("spam.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    return fh


class DummyFMStep(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(name="DummyForwardModel", command=["touch", "dummy.out"])


@plugin(name="dummy")
def installable_forward_model_steps():
    return [DummyFMStep]
