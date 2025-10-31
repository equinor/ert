import logging
from io import StringIO

from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from ert import ErtScript, ForwardModelStepPlugin, plugin


@plugin(name="dummy")
def help_links():
    return {"test": "test", "test2": "test"}


@plugin(name="dummy")
def forward_model_configuration():
    return {"FLOW": {"mpipath": "/foo"}}


@plugin(name="dummy")
def ecl100_config_path():
    return "dummy/path/ecl100_config.yml"


@plugin(name="dummy")
def ecl300_config_path():
    return "dummy/path/ecl300_config.yml"


@plugin(name="dummy")
def flow_config_path():
    return "dummy/path/flow_config.yml"


@plugin(name="dummy")
def installable_jobs():
    return {"job1": "dummy/path/job1", "job2": "dummy/path/job2"}


@plugin(name="dummy")
def installable_workflow_jobs():
    return {"wf_job1": "dummy/path/wf_job1", "wf_job2": "dummy/path/wf_job2"}


@plugin(name="dummy")
def legacy_ertscript_workflow(config):
    class Test(ErtScript):
        def run(self):
            pass

    config.add_workflow(Test)


@plugin(name="dummy")
def site_configurations():
    return {
        "queue_options": {
            "name": "lsf",
            "max_running": "1",
            "submit_sleep": "1",
        },
        "environment_variables": {
            "OMP_NUM_THREADS": "5",
            "MKL_NUM_THREADS": "5",
            "NUMEXPR_NUM_THREADS": "5",
        },
    }


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


span_output = StringIO()
span_processor = BatchSpanProcessor(ConsoleSpanExporter(out=span_output))


@plugin(name="dummy")
def add_span_processor():
    return span_processor


class DummyFMStep(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(name="DummyForwardModel", command=["touch", "dummy.out"])


@plugin(name="dummy")
def installable_forward_model_steps():
    return [DummyFMStep]
