import asyncio
import contextlib
from functools import partial
import importlib
import pathlib
import tempfile
import threading
from pathlib import Path
from typing import Callable, ContextManager, Type

import cloudpickle
import pytest

import ert
import ert_shared.ensemble_evaluator.ensemble.builder as ee
from ensemble_evaluator_utils import _mock_ws
from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig


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
    asyncio.get_event_loop().run_until_complete(transmitter.transmit_record(record))
    return transmitter


@pytest.fixture()
def input_transmitter_factory(transmitter_factory):
    def make_transmitter(name, data):
        return create_input_transmitter(data, transmitter_factory(name))

    return make_transmitter


def create_coefficient_transmitters(
    coefficients, transmitter_factory: Callable[[str], Type[ert.data.RecordTransmitter]]
):
    transmitters = {}
    record_name = "coeffs"
    for iens, values in enumerate(coefficients):
        transmitter = transmitter_factory(record_name)
        transmitters[iens] = {
            record_name: create_input_transmitter(values, transmitter)
        }
    return transmitters


def create_script_transmitter(name: str, location: Path, transmitter_factory):
    script_transmitter = transmitter_factory(name)
    asyncio.get_event_loop().run_until_complete(
        script_transmitter.transmit_file(location, mime="application/octet-stream")
    )
    return script_transmitter


def get_output_transmitters(
    names, transmitter_factory: Callable[[str], Type[ert.data.RecordTransmitter]]
):
    return {name: transmitter_factory(name) for name in names}


def step_output_transmitters(
    step, transmitter_factory: Callable[[str], Type[ert.data.RecordTransmitter]]
):
    transmitters = {}
    for output in step.get_outputs():
        transmitters[output.get_name()] = transmitter_factory(output.get_name())

    return transmitters


@pytest.fixture()
def step_output_transmitters_factory(transmitter_factory):
    return partial(step_output_transmitters, transmitter_factory=transmitter_factory)


def get_poly_scripts(script_base_path, transmitter_factory):
    return {
        "generate_zero_degree": create_script_transmitter(
            "generate_zero_degree",
            script_base_path / "evaluate_coeffs.py",
            transmitter_factory=transmitter_factory,
        ),
        "generate_first_degree": create_script_transmitter(
            "generate_first_degree",
            script_base_path / "evaluate_coeffs.py",
            transmitter_factory=transmitter_factory,
        ),
        "generate_second_degree": create_script_transmitter(
            "generate_second_degree",
            script_base_path / "evaluate_coeffs.py",
            transmitter_factory=transmitter_factory,
        ),
        "sum_up": create_script_transmitter(
            "sum_up",
            script_base_path / "sum_coeffs.py",
            transmitter_factory=transmitter_factory,
        ),
    }


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


def get_degree_step(degree, degree_spelled):
    step_builder = (
        ee.create_step_builder().set_name(f"{degree_spelled}_degree").set_type("unix")
    )

    step_builder.add_input(
        ee.create_file_io_builder()
        .set_name(f"generate_{degree_spelled}_degree")
        .set_path(Path("evaluate_coeffs.py"))
        .set_mime("text/x-python")
        .set_transformation(ert.data.ExecutableRecordTransformation())
    )

    step_builder.add_input(
        ee.create_file_io_builder()
        .set_name("coeffs")
        .set_path(Path("coeffs.json"))
        .set_mime("application/json")
    )

    step_builder.add_output(
        ee.create_file_io_builder()
        .set_name(f"input{degree}")
        .set_path(Path(f"poly_{degree}.out"))
        .set_mime("application/json")
    )
    step_builder.add_job(
        ee.create_job_builder()
        .set_name(f"generate_{degree_spelled}_degree")
        .set_executable(Path("evaluate_coeffs.py"))
        .set_args([f"{degree}"])
    )
    return step_builder


@pytest.fixture()
def zero_degree_step():
    step_builder = get_degree_step(degree=0, degree_spelled="zero")
    step_builder.add_input(
        ee.create_file_io_builder()
        .set_name("input2")
        .set_path(Path("poly_2.out"))
        .set_mime("application/json")
    )
    return step_builder


@pytest.fixture()
def first_degree_step():
    return get_degree_step(degree=1, degree_spelled="first")


@pytest.fixture()
def second_degree_step():
    return get_degree_step(degree=2, degree_spelled="second")


@pytest.fixture()
def sum_coeffs_step():
    step_builder = ee.create_step_builder().set_name("add_coeffs").set_type("unix")

    step_builder.add_input(
        ee.create_file_io_builder()
        .set_name("sum_up")
        .set_path(Path("sum_coeffs.py"))
        .set_mime("text/x-python")
        .set_transformation(ert.data.ExecutableRecordTransformation())
    )

    step_builder.add_input(
        ee.create_file_io_builder()
        .set_name("input0")
        .set_path(Path("poly_0.out"))
        .set_mime("application/json")
    )
    step_builder.add_input(
        ee.create_file_io_builder()
        .set_name("input1")
        .set_path(Path("poly_1.out"))
        .set_mime("application/json")
    )
    step_builder.add_input(
        ee.create_file_io_builder()
        .set_name("input2")
        .set_path(Path("poly_2.out"))
        .set_mime("application/json")
    )

    step_builder.add_output(
        ee.create_file_io_builder()
        .set_name("sum_output")
        .set_path(Path("poly_sum.out"))
        .set_mime("application/json")
    )
    step_builder.add_job(
        ee.create_job_builder()
        .set_name("sum_up")
        .set_executable(Path("sum_coeffs.py"))
        .set_args([])
    )
    return step_builder


@pytest.fixture()
def real_builder(
    sum_coeffs_step, zero_degree_step, first_degree_step, second_degree_step
):
    real_builder = ee.create_realization_builder().active(True)
    real_builder.add_step(sum_coeffs_step)
    real_builder.add_step(zero_degree_step)
    real_builder.add_step(first_degree_step)
    real_builder.add_step(second_degree_step)
    return real_builder


@pytest.fixture()
def poly_ensemble_inputs(coefficients, test_data_path, transmitter_factory):
    script_transmitters = get_poly_scripts(
        script_base_path=test_data_path, transmitter_factory=transmitter_factory
    )

    coeffs_trans = create_coefficient_transmitters(coefficients, transmitter_factory)

    inputs = {}
    for iens in range(len(coeffs_trans)):
        inputs[iens] = {**coeffs_trans[iens], **script_transmitters}

    return inputs


@pytest.fixture()
def poly_ensemble_outputs(transmitter_factory, ensemble_size):
    outputs = {
        i: get_output_transmitters(
            ["sum_output", "input0", "input1", "input2"], transmitter_factory
        )
        for i in range(ensemble_size)
    }

    return outputs


@pytest.fixture()
def poly_ensemble_builder(
    real_builder, poly_ensemble_inputs, poly_ensemble_outputs, ensemble_size
):
    builder = (
        ee.create_ensemble_builder()
        .set_custom_port_range(custom_port_range=range(1024, 65535))
        .set_ensemble_size(ensemble_size)
        .set_max_running(6)
        .set_max_retries(2)
        .set_executor("local")
        .set_forward_model(real_builder)
        .set_inputs(poly_ensemble_inputs)
        .set_outputs(poly_ensemble_outputs)
    )
    return builder


@pytest.fixture()
def poly_ensemble(poly_ensemble_builder):
    return poly_ensemble_builder.build()


def get_function_ensemble(pickled_function):
    step_builder = (
        ee.create_step_builder().set_name("function_evaluation").set_type("function")
    )

    step_builder.add_input(
        ee.create_file_io_builder()
        .set_name("coeffs")
        .set_path("coeffs")
        .set_mime("application/json")
    )

    step_builder.add_output(
        ee.create_file_io_builder()
        .set_name("function_output")
        .set_path("output")
        .set_mime("application/json")
    )
    step_builder.add_job(
        ee.create_job_builder()
        .set_name("user_defined_function")
        .set_executable(pickled_function)
    )
    real_builder = ee.create_realization_builder().active(True).add_step(step_builder)

    builder = (
        ee.create_ensemble_builder()
        .set_custom_port_range(custom_port_range=range(1024, 65535))
        .set_ensemble_size(25)
        .set_max_running(6)
        .set_max_retries(2)
        .set_executor("local")
        .set_forward_model(real_builder)
    )
    return builder


@pytest.fixture()
def function_ensemble_inputs(coefficients, transmitter_factory):
    coeffs_trans = create_coefficient_transmitters(coefficients, transmitter_factory)
    return coeffs_trans


@pytest.fixture()
def function_ensemble_outputs(transmitter_factory, ensemble_size):
    outputs = {
        i: get_output_transmitters(["function_output"], transmitter_factory)
        for i in range(ensemble_size)
    }

    return outputs


@pytest.fixture()
def function_ensemble_builder_factory(
    function_ensemble_inputs, function_ensemble_outputs, ensemble_size
):
    job_builder = ee.create_job_builder().set_name("user_defined_function")

    step_builder = (
        ee.create_step_builder().set_name("function_evaluation").set_type("function")
    )

    step_builder.add_input(
        ee.create_file_io_builder()
        .set_name("coeffs")
        .set_path("coeffs")
        .set_mime("application/json")
    )

    step_builder.add_output(
        ee.create_file_io_builder()
        .set_name("function_output")
        .set_path("output")
        .set_mime("application/json")
    )
    step_builder.add_job(job_builder)
    real_builder = ee.create_realization_builder().active(True).add_step(step_builder)

    builder = (
        ee.create_ensemble_builder()
        .set_custom_port_range(custom_port_range=range(1024, 65535))
        .set_ensemble_size(ensemble_size)
        .set_max_running(6)
        .set_max_retries(2)
        .set_executor("local")
        .set_forward_model(real_builder)
        .set_inputs(function_ensemble_inputs)
        .set_outputs(function_ensemble_outputs)
    )

    def build(pickled_function):
        job_builder.set_executable(pickled_function)
        return builder

    return build


@pytest.fixture()
def evaluator_config(unused_tcp_port):
    fixed_port = range(unused_tcp_port, unused_tcp_port)
    return EvaluatorServerConfig(custom_port_range=fixed_port)


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
        # Create temporary module that defines a function `bar`
        # 'bar' returns a call to different function 'internal_call' defined in the same python file
        module_path = Path(tmpdir) / "foo"
        module_path.mkdir()
        init_file = module_path / "__init__.py"
        init_file.touch()
        file_path = module_path / "bar.py"
        file_path.write_text(
            "def bar(coeffs):\n    return internal_call(coeffs)\n"
            "def internal_call(coeffs):\n    return [sum(coeffs.values())]\n"
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
            import foo.bar

        # Make sure the function is no longer available before we start creating the flow and task
        del func

        return pickle_func
