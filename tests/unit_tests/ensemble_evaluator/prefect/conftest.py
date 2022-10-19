import contextlib
import importlib
import pathlib
import tempfile
import threading
from functools import partial
from pathlib import Path
from typing import Callable, ContextManager, Type

import cloudpickle
import pytest

import ert
import ert.ensemble_evaluator as ee
from _ert_job_runner.client import Client
from ert.async_utils import get_event_loop
from ert.ensemble_evaluator.config import EvaluatorServerConfig

from ..ensemble_evaluator_utils import _mock_ws


@contextlib.contextmanager
def shared_disk_factory_context(
    **kwargs,
) -> ContextManager[Callable[[str], Type[ert.data.RecordTransmitter]]]:
    tmp_path = tempfile.TemporaryDirectory()
    tmp_storage_path = pathlib.Path(tmp_path.name) / ".shared-storage"
    tmp_storage_path.mkdir(parents=True)

    def shared_disk_factory(name: str) -> ert.data.SharedDiskRecordTransmitter:
        return ert.data.SharedDiskRecordTransmitter(
            name=name,
            storage_path=tmp_storage_path,
        )

    try:
        yield shared_disk_factory
    finally:
        tmp_path.cleanup()


transmitter_factory_context: ContextManager[
    Callable[[str], Type[ert.data.RecordTransmitter]]
] = shared_disk_factory_context


def create_input_transmitter(data, transmitter: Type[ert.data.RecordTransmitter]):
    record = ert.data.NumericalRecord(data=data)
    get_event_loop().run_until_complete(transmitter.transmit_record(record))
    return transmitter


@pytest.fixture()
def input_transmitter_factory(transmitter_factory):
    def make_transmitter(name, data):
        return create_input_transmitter(data, transmitter_factory(name))

    return make_transmitter


def create_script_transmitter(name: str, location: Path, transmitter_factory):
    async def transform_output(transmitter, location):
        transformation = ert.data.ExecutableTransformation(location=location)
        record = await transformation.to_record()
        await transmitter.transmit_record(record)

    script_transmitter = transmitter_factory(name)
    get_event_loop().run_until_complete(
        transform_output(script_transmitter, location=location)
    )
    return script_transmitter


@pytest.fixture()
def transmitter_factory():
    with transmitter_factory_context() as transmitter_factory:
        yield transmitter_factory


@pytest.fixture()
def ensemble_size():
    return 2


@pytest.fixture()
def coefficients():
    return [{"a": a, "b": b, "c": c} for (a, b, c) in [(1, 2, 3), (4, 2, 1)]]


@pytest.fixture()
def test_data_path():
    return Path(__file__).parent / "scripts"


def get_degree_step(
    degree, degree_spelled, transmitter_factory, test_data_path, coefficients
):
    step_builder = (
        ee.StepBuilder().set_name(f"{degree_spelled}_degree").set_type("unix")
    )

    input_name = f"generate_{degree_spelled}_degree"
    step_builder.add_input(
        ee.InputBuilder()
        .set_name(input_name)
        .set_transformation(
            ert.data.ExecutableTransformation(
                location=Path("evaluate_coeffs.py"), mime="text/x-python"
            )
        )
        .set_transmitter_factory(
            partial(
                create_script_transmitter,
                input_name,
                test_data_path / "evaluate_coeffs.py",
                transmitter_factory=transmitter_factory,
            )
        )
    )

    coeffs_input = (
        ee.InputBuilder()
        .set_name("coeffs")
        .set_transformation(
            ert.data.SerializationTransformation(
                location=Path("coeffs.json"), mime="application/json"
            )
        )
    )
    for iens, values in enumerate(coefficients):
        transmitter = create_input_transmitter(values, transmitter_factory("coeffs"))
        coeffs_input.set_transmitter_factory(lambda _t=transmitter: _t, iens)
    step_builder.add_input(coeffs_input)

    output_name = f"input{degree}"
    step_builder.add_output(
        ee.OutputBuilder()
        .set_name(output_name)
        .set_transformation(
            ert.data.SerializationTransformation(
                location=Path(f"poly_{degree}.out"), mime="application/json"
            )
        )
        .set_transmitter_factory(partial(transmitter_factory, output_name))
    )
    step_builder.add_job(
        ee.JobBuilder()
        .set_index("0")
        .set_name(f"generate_{degree_spelled}_degree")
        .set_executable(Path("evaluate_coeffs.py"))
        .set_args([f"{degree}"])
    )
    return step_builder


@pytest.fixture()
def zero_degree_step(transmitter_factory, test_data_path, coefficients):
    step_builder = get_degree_step(
        degree=0,
        degree_spelled="zero",
        transmitter_factory=transmitter_factory,
        test_data_path=test_data_path,
        coefficients=coefficients,
    )

    step_builder.add_input(
        ee.InputBuilder()
        .set_name("input2")
        .set_transformation(
            ert.data.SerializationTransformation(
                location=Path("poly_2.out"), mime="application/json"
            )
        )
        .set_transmitter_factory(partial(transmitter_factory, "input2"))
    )
    return step_builder


@pytest.fixture()
def first_degree_step(transmitter_factory, test_data_path, coefficients):
    return get_degree_step(
        degree=1,
        degree_spelled="first",
        transmitter_factory=transmitter_factory,
        test_data_path=test_data_path,
        coefficients=coefficients,
    )


@pytest.fixture()
def second_degree_step(transmitter_factory, test_data_path, coefficients):
    return get_degree_step(
        degree=2,
        degree_spelled="second",
        transmitter_factory=transmitter_factory,
        test_data_path=test_data_path,
        coefficients=coefficients,
    )


@pytest.fixture()
def sum_coeffs_step(test_data_path, transmitter_factory):
    step_builder = ee.StepBuilder().set_name("add_coeffs").set_type("unix")
    step_builder.add_input(
        ee.InputBuilder()
        .set_name("sum_up")
        .set_transformation(
            ert.data.ExecutableTransformation(
                location=Path("sum_coeffs.py"), mime="text/x-python"
            )
        )
        .set_transmitter_factory(
            partial(
                create_script_transmitter,
                "sum_up",
                test_data_path / "sum_coeffs.py",
                transmitter_factory=transmitter_factory,
            )
        )
    )

    step_builder.add_input(
        ee.InputBuilder()
        .set_name("input0")
        .set_transformation(
            ert.data.SerializationTransformation(
                location=Path("poly_0.out"), mime="application/json"
            )
        )
        .set_transmitter_factory(partial(transmitter_factory, "input0"))
    )

    step_builder.add_input(
        ee.InputBuilder()
        .set_name("input1")
        .set_transformation(
            ert.data.SerializationTransformation(
                location=Path("poly_1.out"), mime="application/json"
            )
        )
        .set_transmitter_factory(partial(transmitter_factory, "input1"))
    )

    step_builder.add_input(
        ee.InputBuilder()
        .set_name("input2")
        .set_transformation(
            ert.data.SerializationTransformation(
                location=Path("poly_2.out"), mime="application/json"
            )
        )
        .set_transmitter_factory(partial(transmitter_factory, "input2"))
    )

    step_builder.add_output(
        ee.OutputBuilder()
        .set_name("sum_output")
        .set_transformation(
            ert.data.SerializationTransformation(
                location=Path("poly_sum.out"), mime="application/json"
            )
        )
        .set_transmitter_factory(partial(transmitter_factory, "sum_output"))
    )

    step_builder.add_job(
        ee.JobBuilder()
        .set_index("0")
        .set_name("sum_up")
        .set_executable(Path("sum_coeffs.py"))
        .set_args([])
    )
    return step_builder


@pytest.fixture()
def real_builder(
    sum_coeffs_step, zero_degree_step, first_degree_step, second_degree_step
):
    real_builder = ee.RealizationBuilder().active(True)
    real_builder.add_step(sum_coeffs_step)
    real_builder.add_step(zero_degree_step)
    real_builder.add_step(first_degree_step)
    real_builder.add_step(second_degree_step)
    return real_builder


@pytest.fixture()
def poly_ensemble_builder(real_builder, ensemble_size):
    builder = (
        ee.EnsembleBuilder()
        .set_custom_port_range(custom_port_range=range(1024, 65535))
        .set_ensemble_size(ensemble_size)
        .set_max_running(6)
        .set_max_retries(2)
        .set_executor("local")
        .set_forward_model(real_builder)
        .set_id("0")
    )
    return builder


@pytest.fixture()
def poly_ensemble(poly_ensemble_builder):
    ens = poly_ensemble_builder.build()
    return ens


@pytest.fixture()
def function_ensemble_builder_factory(
    ensemble_size,
    transmitter_factory,
    coefficients,
):
    job_builder = ee.JobBuilder().set_name("user_defined_function").set_index("0")

    step_builder = ee.StepBuilder().set_name("function_evaluation").set_type("function")

    coeffs_input = (
        ee.InputBuilder()
        .set_name("coeffs")
        .set_transformation(
            ert.data.SerializationTransformation(
                location=Path("coeffs"), mime="application/json"
            )
        )
    )

    for iens, values in enumerate(coefficients):
        transmitter = create_input_transmitter(values, transmitter_factory("coeffs"))
        coeffs_input.set_transmitter_factory(lambda _t=transmitter: _t, iens)
    step_builder.add_input(coeffs_input)

    step_builder.add_output(
        ee.OutputBuilder()
        .set_name("function_output")
        .set_transformation(
            ert.data.SerializationTransformation(
                location="output", mime="application/json"
            )
        )
        .set_transmitter_factory(partial(transmitter_factory, "function_output"))
    )
    step_builder.add_job(job_builder)
    real_builder = ee.RealizationBuilder().active(True).add_step(step_builder)

    builder = (
        ee.EnsembleBuilder()
        .set_custom_port_range(custom_port_range=range(1024, 65535))
        .set_ensemble_size(ensemble_size)
        .set_max_running(6)
        .set_max_retries(2)
        .set_executor("local")
        .set_forward_model(real_builder)
        .set_id("0")
    )

    def build(pickled_function):
        job_builder.set_executable(pickled_function)
        return builder

    return build


@pytest.fixture()
def evaluator_config():
    return EvaluatorServerConfig(
        custom_port_range=range(1024, 65535),
        custom_host="127.0.0.1",
        use_token=False,
        generate_cert=False,
    )


class MockWSMonitor:
    def __init__(self, unused_tcp_port) -> None:
        self.host = "localhost"
        self.url = f"ws://{self.host}:{unused_tcp_port}"
        self.messages = []
        self.mock_ws_thread = threading.Thread(
            target=partial(_mock_ws, messages=self.messages),
            args=(self.host, unused_tcp_port),
        )

    def start(self):
        self.mock_ws_thread.start()

    def join(self):
        with Client(self.url) as c:
            c.send("stop")
        self.mock_ws_thread.join()

    def join_and_get_messages(self):
        self.join()
        return self.messages


@pytest.fixture()
def mock_ws_monitor(unused_tcp_port):
    mock_ws_monitor = MockWSMonitor(unused_tcp_port=unused_tcp_port)
    mock_ws_monitor.start()
    yield mock_ws_monitor


@pytest.fixture()
def external_sum_function(tmpdir):
    with tmpdir.as_cwd():
        # Create temporary module that defines a function 'bar'. 'bar' returns a
        # call to a different function 'internal_call', defined in the same
        # python file.
        module_path = Path(tmpdir) / "foo"
        module_path.mkdir()
        init_file = module_path / "__init__.py"
        init_file.touch()
        file_path = module_path / "bar.py"
        file_path.write_text(
            "def bar(coeffs):\n"
            "    return internal_call(coeffs)\n"
            "def internal_call(coeffs):\n"
            "    return {'function_output':[sum(coeffs.values())]}\n"
        )
        spec = importlib.util.spec_from_file_location("foo", str(file_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, "bar")
        pickle_func = cloudpickle.dumps(func)
        init_file.unlink()
        file_path.unlink()

        # Check module is not in the python environment
        with pytest.raises(ModuleNotFoundError):
            # pylint: disable=unused-import,import-error
            import foo.bar

        # Make sure the function is no longer available before we start creating
        # the flow and task
        del func

        return pickle_func
