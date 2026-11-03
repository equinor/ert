import logging
from io import StringIO

from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from ert import EcosystemPlugin, ErtScript, ForwardModelStepPlugin, plugin


@plugin(name="dummy")
def help_links():
    return {"test": "test", "test2": "test"}


@plugin(name="dummy")
def forward_model_configuration():
    return {"FLOW": {"mpipath": "/foo"}}


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


PLUGIN_IP_ADDRESS = "1.0.0.127"


@plugin(name="dummy")
def get_ip_address():
    return PLUGIN_IP_ADDRESS


class DummyEcosystem(EcosystemPlugin):
    name: str = "DUMMY_ECOSYSTEM"

    def configure(self, config_dict):
        config_dict.setdefault("FORWARD_MODEL", []).append(["DummyForwardModel"])


@plugin(name="dummy")
def ecosystem_plugin():
    return DummyEcosystem
