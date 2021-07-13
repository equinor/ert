import asyncio
import copy
import importlib
import os
import os.path
import pickle
import sys
import threading
from collections import defaultdict
from datetime import timedelta
from functools import partial
from itertools import permutations
from pathlib import Path
from typing import Set

import cloudpickle
import prefect
import pytest
import yaml
from ensemble_evaluator_utils import _mock_ws
from ert_utils import tmp
from prefect import Flow

import ert
import ert_shared.ensemble_evaluator.ensemble.builder as ee
from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.status.entity import state
from ert_shared.ensemble_evaluator.entity.unix_step import UnixTask
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.ensemble.prefect import PrefectEnsemble


def parse_config(path):
    conf_path = Path(path).resolve()
    with open(conf_path, "r") as f:
        config = yaml.safe_load(f)
    for step in config.get(ids.STEPS):
        for input_ in step.get(ids.INPUTS):
            input_[ids.LOCATION] = Path(input_[ids.LOCATION])
        for job in step.get(ids.JOBS):
            job[ids.EXECUTABLE] = (
                Path(job[ids.EXECUTABLE])
                if step.get(ids.TYPE) == ids.UNIX
                else job[ids.EXECUTABLE]
            )
    return config


def input_transmitter(name, data, storage_path):
    transmitter = ert.data.SharedDiskRecordTransmitter(
        name=name, storage_path=Path(storage_path)
    )
    asyncio.get_event_loop().run_until_complete(transmitter.transmit_data(data))
    return {name: transmitter}


def coefficient_transmitters(coefficients, storage_path):
    transmitters = {}
    record_name = "coeffs"
    for iens, values in enumerate(coefficients):
        transmitters[iens] = input_transmitter(record_name, values, storage_path)
    return transmitters


def script_transmitters(config):
    transmitters = defaultdict(dict)
    for step in config.get(ids.STEPS):
        for job in step.get(ids.JOBS):
            for iens in range(config.get(ids.REALIZATIONS)):
                transmitters[iens].update(
                    script_transmitter(
                        name=job.get(ids.NAME),
                        location=Path(job.get(ids.EXECUTABLE)),
                        storage_path=config.get(ids.STORAGE)["storage_path"],
                    )
                )
    return dict(transmitters)


def script_transmitter(name, location, storage_path):
    transmitter = ert.data.SharedDiskRecordTransmitter(
        name=name, storage_path=Path(storage_path)
    )
    with open(location, "rb") as f:
        asyncio.get_event_loop().run_until_complete(
            transmitter.transmit_data([f.read()])
        )

    return {name: transmitter}


def output_transmitters(config):
    tmp_input_folder = "output_files"
    os.makedirs(tmp_input_folder)
    transmitters = defaultdict(dict)
    for step in config.get(ids.STEPS):
        for output in step.get(ids.OUTPUTS):
            for iens in range(config.get(ids.REALIZATIONS)):
                transmitters[iens][
                    output.get(ids.RECORD)
                ] = ert.data.SharedDiskRecordTransmitter(
                    output.get(ids.RECORD),
                    storage_path=Path(config.get(ids.STORAGE)["storage_path"]),
                )
    return dict(transmitters)


def step_output_transmitters(step, storage_path):
    transmitters = {}
    for output in step.get_outputs():
        transmitters[output.get_name()] = ert.data.SharedDiskRecordTransmitter(
            name=output.get_name(), storage_path=Path(storage_path)
        )

    return transmitters


@pytest.fixture()
def coefficients():
    return [{"a": a, "b": b, "c": c} for (a, b, c) in [(1, 2, 3), (4, 2, 1)]]


@pytest.fixture()
def function_config(tmpdir):
    storage = {"type": "shared_disk", "storage_path": ".my_storage"}
    inputs = [
        {
            "record": "coeffs",
            "location": "coeffs",
            "mime": "application/json",
            "is_executable": False,
        }
    ]
    outputs = [
        {
            "record": "function_output",
            "location": "output",
            "mime": "application/json",
        }
    ]
    jobs = [
        {
            "name": "user_defined_function",
            "executable": "place_holder",
            "output": "output",
        }
    ]
    steps = [
        {
            "name": "function_evaluation",
            "type": "function",
            "inputs": inputs,
            "outputs": outputs,
            "jobs": jobs,
        }
    ]
    config = {}
    config["max_running"] = 1
    config["storage"] = storage
    config["steps"] = steps
    return config


def get_step(step_name, inputs, outputs, jobs, type_="unix"):
    step_source = "/ert/ee/test_ee_id/real/0/step/0"
    step_builder = ee.create_step_builder()
    for idx, (name, executable, args) in enumerate(jobs):
        step_builder.add_job(
            ee.create_job_builder()
            .set_id(str(idx))
            .set_name(name)
            .set_executable(executable)
            .set_args(args)
            .set_step_source(step_source)
        )
    step_builder.set_source(source=step_source)
    step_builder.set_id("0")
    step_builder.set_name(step_name)
    for name, path, mime in inputs:
        step_builder.add_input(
            ee.create_file_io_builder()
            .set_name(name)
            .set_path(Path(path))
            .set_mime(mime)
            .set_executable()
        )
    for name, path, mime in outputs:
        step_builder.add_output(
            ee.create_file_io_builder().set_name(name).set_path(path).set_mime(mime)
        )
    step_builder.set_type(type_)
    return step_builder.build()


def test_get_flow(coefficients, unused_tcp_port, source_root):
    with tmp(source_root / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        config.update(
            {
                "config_path": os.getcwd(),
                ids.REALIZATIONS: 2,
                ids.EXECUTOR: "local",
            }
        )
        inputs = {}
        coeffs_trans = coefficient_transmitters(
            coefficients, config.get(ids.STORAGE)["storage_path"]
        )
        script_trans = script_transmitters(config)
        for iens in range(2):
            inputs[iens] = {**coeffs_trans[iens], **script_trans[iens]}
        config.update(
            {
                "inputs": inputs,
                "outputs": output_transmitters(config),
            }
        )
        fixed_port = range(unused_tcp_port, unused_tcp_port)
        custom_range = range(1024, 65535)
        server_config = EvaluatorServerConfig(custom_port_range=fixed_port)
        for permuted_steps in permutations(config["steps"]):
            permuted_config = copy.deepcopy(config)
            permuted_config["steps"] = permuted_steps
            permuted_config["dispatch_uri"] = server_config.dispatch_uri
            ensemble = PrefectEnsemble(permuted_config, custom_port_range=custom_range)

            for iens in range(2):
                with prefect.context(
                    url=server_config.url,
                    token=server_config.token,
                    cert=server_config.cert,
                ):
                    flow = ensemble.get_flow(ensemble._ee_id, [iens])

                # Get the ordered tasks and retrieve their step ids.
                flow_steps = [
                    task.get_step()
                    for task in flow.sorted_tasks()
                    if isinstance(task, UnixTask)
                ]
                assert len(flow_steps) == 4

                realization_steps = list(
                    ensemble.get_reals()[iens].get_steps_sorted_topologically()
                )

                # Testing realization steps
                for step_ordering in [realization_steps, flow_steps]:
                    mapping = {
                        step._name: idx for idx, step in enumerate(step_ordering)
                    }
                    assert mapping["second_degree"] < mapping["zero_degree"]
                    assert mapping["zero_degree"] < mapping["add_coeffs"]
                    assert mapping["first_degree"] < mapping["add_coeffs"]
                    assert mapping["second_degree"] < mapping["add_coeffs"]


def test_unix_task(unused_tcp_port, tmpdir, source_root):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    script_location = (
        source_root / "test-data/local/prefect_test_case/unix_test_script.py"
    )
    input_ = script_transmitter("script", script_location, storage_path=tmpdir)
    step = get_step(
        step_name="test_step",
        inputs=[("script", Path("unix_test_script.py"), "application/x-python")],
        outputs=[("output", Path("output.out"), "application/json")],
        jobs=[("script", Path("unix_test_script.py"), ["vas"])],
        type_="unix",
    )

    with prefect.context(url=url, token=None, cert=None):
        output_trans = step_output_transmitters(step, storage_path=tmpdir)
        with Flow("testing") as flow:
            task = step.get_task(output_transmitters=output_trans, ee_id="test_ee_id")
            result = task(inputs=input_)
        with tmp():
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    with Client(url) as c:
        c.send("stop")
    mock_ws_thread.join()

    task_result = flow_run.result[result]
    assert task_result.is_successful()
    assert flow_run.is_successful()

    assert len(task_result.result) == 1
    expected_uri = output_trans["output"]._uri
    output_uri = task_result.result["output"]._uri
    assert expected_uri == output_uri


def test_function_step(unused_tcp_port, tmpdir):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    test_values = {"values": [42, 24, 6]}
    inputs = input_transmitter("values", test_values["values"], storage_path=tmpdir)

    def sum_function(values):
        return [sum(values)]

    step = get_step(
        step_name="test_step",
        inputs=[("values", "NA", "text/whatever")],
        outputs=[("output", Path("output.out"), "application/json")],
        jobs=[("test_function", cloudpickle.dumps(sum_function), None)],
        type_="function",
    )

    with prefect.context(url=url, token=None, cert=None):
        output_trans = step_output_transmitters(step, storage_path=tmpdir)
        with Flow("testing") as flow:
            task = step.get_task(output_transmitters=output_trans, ee_id="test_ee_id")
            result = task(inputs=inputs)
        with tmp():
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    with Client(url) as c:
        c.send("stop")
    mock_ws_thread.join()

    task_result = flow_run.result[result]
    assert task_result.is_successful()
    assert flow_run.is_successful()

    assert len(task_result.result) == 1
    expected_uri = output_trans["output"]._uri
    output_uri = task_result.result["output"]._uri
    assert expected_uri == output_uri
    transmitted_record = asyncio.get_event_loop().run_until_complete(
        task_result.result["output"].load()
    )
    transmitted_result = transmitted_record.data
    expected_result = sum_function(**test_values)
    assert expected_result == transmitted_result


def test_function_step_for_function_defined_outside_py_environment(
    unused_tcp_port, tmpdir
):
    # Create temporary module that defines a function `bar`
    # 'bar' returns a call to different function 'internal_call' defined in the same python file
    with tmpdir.as_cwd():
        module_path = Path(tmpdir) / "foo"
        module_path.mkdir()
        init_file = module_path / "__init__.py"
        init_file.touch()
        file_path = module_path / "bar.py"
        file_path.write_text(
            "def bar(values):\n    return internal_call(values)\n"
            "def internal_call(values):\n    return [sum(values)]\n"
        )
        spec = importlib.util.spec_from_file_location("foo", str(file_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, "bar")

    # Check module is not in the python environment
    with pytest.raises(ModuleNotFoundError):
        import foo.bar

    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    test_values = {"values": [42, 24, 6]}
    inputs = input_transmitter("values", test_values["values"], storage_path=tmpdir)

    step = get_step(
        step_name="test_step",
        inputs=[("values", "NA", "text/whatever")],
        outputs=[("output", Path("output.out"), "application/json")],
        jobs=[("test_function", cloudpickle.dumps(func), None)],
        type_="function",
    )
    expected_result = func(**test_values)
    # Make sure the function is no longer available before we start creating the flow and task
    del func

    with prefect.context(url=url, token=None, cert=None):
        output_trans = step_output_transmitters(step, storage_path=tmpdir)
        with Flow("testing") as flow:
            task = step.get_task(output_transmitters=output_trans, ee_id="test_ee_id")
            result = task(inputs=inputs)
        with tmp():
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    with Client(url) as c:
        c.send("stop")
    mock_ws_thread.join()

    task_result = flow_run.result[result]
    assert task_result.is_successful()
    assert flow_run.is_successful()

    assert len(task_result.result) == 1
    expected_uri = output_trans["output"]._uri
    output_uri = task_result.result["output"]._uri
    assert expected_uri == output_uri
    transmitted_record = asyncio.get_event_loop().run_until_complete(
        task_result.result["output"].load()
    )
    transmitted_result = transmitted_record.data
    assert expected_result == transmitted_result


def test_unix_step_error(unused_tcp_port, tmpdir, source_root):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    script_location = (
        source_root / "test-data/local/prefect_test_case/unix_test_script.py"
    )
    input_ = script_transmitter("test_script", script_location, storage_path=tmpdir)
    step = get_step(
        step_name="test_step",
        inputs=[("test_script", Path("unix_test_script.py"), "application/x-python")],
        outputs=[("output", Path("output.out"), "application/json")],
        jobs=[("test_script", Path("unix_test_script.py"), ["foo", "bar"])],
        type_="unix",
    )

    with prefect.context(url=url, token=None, cert=None):
        output_trans = step_output_transmitters(step, storage_path=tmpdir)
        with Flow("testing") as flow:
            task = step.get_task(output_transmitters=output_trans, ee_id="test_ee_id")
            result = task(inputs=input_)
        with tmp():
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    with Client(url) as c:
        c.send("stop")
    mock_ws_thread.join()

    task_result = flow_run.result[result]
    assert not task_result.is_successful()
    assert not flow_run.is_successful()

    assert isinstance(task_result.result, Exception)
    assert (
        "unix_test_script.py: error: unrecognized arguments: bar" in task_result.message
    )


class _MockedPrefectEnsemble:
    def __init__(self):
        self._ee_id = "test_ee_id"

    _on_task_failure = PrefectEnsemble._on_task_failure


def test_on_task_failure(unused_tcp_port, tmpdir, source_root):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ensemble = _MockedPrefectEnsemble()

    mock_ws_thread.start()
    script_location = (
        source_root / "test-data/local/prefect_test_case/unix_test_retry_script.py"
    )
    input_ = script_transmitter("script", script_location, storage_path=tmpdir)
    with tmp() as runpath:
        step = get_step(
            step_name="test_step",
            inputs=[
                ("script", Path("unix_test_retry_script.py"), "application/x-python")
            ],
            outputs=[],
            jobs=[("script", Path("unix_test_retry_script.py"), [runpath])],
            type_="unix",
        )

        with prefect.context(url=url, token=None, cert=None):
            output_trans = step_output_transmitters(step, storage_path=tmpdir)
            with Flow("testing") as flow:
                task = step.get_task(
                    output_transmitters=output_trans,
                    ee_id="test_ee_id",
                    max_retries=3,
                    retry_delay=timedelta(seconds=1),
                    on_failure=mock_ensemble._on_task_failure,
                )
                result = task(inputs=input_)
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    with Client(url) as c:
        c.send("stop")
    mock_ws_thread.join()

    task_result = flow_run.result[result]
    assert task_result.is_successful()
    assert flow_run.is_successful()

    fail_job_messages = [msg for msg in messages if ids.EVTYPE_FM_JOB_FAILURE in msg]
    fail_step_messages = [msg for msg in messages if ids.EVTYPE_FM_STEP_FAILURE in msg]

    expected_job_failed_messages = 2
    expected_step_failed_messages = 0
    assert expected_job_failed_messages == len(fail_job_messages)
    assert expected_step_failed_messages == len(fail_step_messages)


def test_on_task_failure_fail_step(unused_tcp_port, tmpdir, source_root):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ensemble = _MockedPrefectEnsemble()

    mock_ws_thread.start()
    script_location = (
        source_root / "test-data/local/prefect_test_case/unix_test_retry_script.py"
    )
    input_ = script_transmitter("script", script_location, storage_path=tmpdir)
    with tmp() as runpath:
        step = get_step(
            step_name="test_step",
            inputs=[
                ("script", Path("unix_test_retry_script.py"), "application/x-python")
            ],
            outputs=[],
            jobs=[("script", Path("unix_test_retry_script.py"), [runpath])],
            type_="unix",
        )

        with prefect.context(url=url, token=None, cert=None):
            output_trans = step_output_transmitters(step, storage_path=tmpdir)
            with Flow("testing") as flow:
                task = step.get_task(
                    output_transmitters=output_trans,
                    ee_id="test_ee_id",
                    max_retries=1,
                    retry_delay=timedelta(seconds=1),
                    on_failure=mock_ensemble._on_task_failure,
                )
                result = task(inputs=input_)
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    with Client(url) as c:
        c.send("stop")
    mock_ws_thread.join()

    task_result = flow_run.result[result]
    assert not task_result.is_successful()
    assert not flow_run.is_successful()

    fail_job_messages = [msg for msg in messages if ids.EVTYPE_FM_JOB_FAILURE in msg]
    fail_step_messages = [msg for msg in messages if ids.EVTYPE_FM_STEP_FAILURE in msg]

    expected_job_failed_messages = 2
    expected_step_failed_messages = 1
    assert expected_job_failed_messages == len(fail_job_messages)
    assert expected_step_failed_messages == len(fail_step_messages)


@pytest.mark.timeout(60)
def test_prefect_retries(unused_tcp_port, coefficients, tmpdir, function_config):
    def function_that_fails_once(coeffs):
        run_path = Path("ran_once")
        if not run_path.exists():
            run_path.touch()
            raise RuntimeError("This is an expected ERROR")
        run_path.unlink()
        return []

    with tmpdir.as_cwd():
        pickle_func = cloudpickle.dumps(function_that_fails_once)
        config = function_config
        coeffs_trans = coefficient_transmitters(
            coefficients, config.get(ids.STORAGE)["storage_path"]
        )

        fixed_port = range(unused_tcp_port, unused_tcp_port)
        custom_range = range(1024, 65535)
        service_config = EvaluatorServerConfig(custom_port_range=fixed_port)
        config["realizations"] = len(coefficients)
        config["executor"] = "local"
        config["max_retries"] = 2
        config["retry_delay"] = 1
        config["steps"][0]["jobs"][0]["executable"] = pickle_func
        config["inputs"] = {
            iens: coeffs_trans[iens] for iens in range(len(coefficients))
        }
        config["outputs"] = output_transmitters(config)
        config["dispatch_uri"] = service_config.dispatch_uri

        ensemble = PrefectEnsemble(config, custom_port_range=custom_range)
        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")
        error_event_reals: Set[str] = set()
        with evaluator.run() as mon:
            for event in mon.track():
                # Capture the job error messages
                if event.data is not None and "This is an expected ERROR" in str(
                    event.data
                ):
                    error_event_reals.update(event.data["reals"].keys())
                if isinstance(event.data, dict) and event.data.get("status") in [
                    "Failed",
                    "Stopped",
                ]:
                    mon.signal_done()
        assert evaluator._ensemble.get_status() == "Stopped"
        successful_realizations = evaluator._ensemble.get_successful_realizations()
        assert successful_realizations == config["realizations"]
        # Check we get only one job error message per realization
        assert len(error_event_reals) == config["realizations"]
        assert "0" in error_event_reals
        assert "1" in error_event_reals


@pytest.mark.timeout(60)
def test_prefect_no_retries(unused_tcp_port, coefficients, tmpdir, function_config):
    def function_that_fails_once(coeffs):
        run_path = Path("ran_once")
        if not run_path.exists():
            run_path.touch()
            raise RuntimeError("This is an expected ERROR")
        run_path.unlink()
        return []

    with tmpdir.as_cwd():
        pickle_func = cloudpickle.dumps(function_that_fails_once)
        config = function_config
        coeffs_trans = coefficient_transmitters(
            coefficients, config.get(ids.STORAGE)["storage_path"]
        )

        fixed_port = range(unused_tcp_port, unused_tcp_port)
        custom_range = range(1024, 65535)

        service_config = EvaluatorServerConfig(custom_port_range=fixed_port)
        config["realizations"] = len(coefficients)
        config["executor"] = "local"
        config["max_retries"] = 0
        config["retry_delay"] = 1
        config["steps"][0]["jobs"][0]["executable"] = pickle_func
        config["inputs"] = {
            iens: coeffs_trans[iens] for iens in range(len(coefficients))
        }
        config["outputs"] = output_transmitters(config)
        config["dispatch_uri"] = service_config.dispatch_uri

        ensemble = PrefectEnsemble(config, custom_port_range=custom_range)
        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")

        event_list = []
        with evaluator.run() as mon:
            for event in mon.track():
                event_list.append(event)
                if event.data is not None and event.data.get("status") in [
                    state.ENSEMBLE_STATE_FAILED,
                    state.ENSEMBLE_STATE_STOPPED,
                ]:
                    mon.signal_done()

        step_failed = False
        job_failed = False
        for real in ensemble.snapshot.get_reals().values():
            for step in real.steps.values():
                for job in step.jobs.values():
                    if job.status == state.JOB_STATE_FAILURE:
                        job_failed = True
                        assert job.error == "This is an expected ERROR"
                        if step.status == state.STEP_STATE_FAILURE:
                            step_failed = True

        assert ensemble.get_status() == state.ENSEMBLE_STATE_FAILED
        assert job_failed, f"Events: {event_list}"
        assert step_failed, f"Events: {event_list}"


def dummy_get_flow(*args, **kwargs):
    raise RuntimeError()


@pytest.mark.timeout(60)
def test_run_prefect_ensemble(unused_tcp_port, coefficients, source_root):
    test_path = source_root / "test-data/local/prefect_test_case"
    with tmp(test_path):
        config = parse_config("config.yml")
        config.update(
            {
                "config_path": os.getcwd(),
                "realizations": 2,
                "executor": "local",
            }
        )
        inputs = {}
        coeffs_trans = coefficient_transmitters(
            coefficients, config.get(ids.STORAGE)["storage_path"]
        )
        script_trans = script_transmitters(config)
        for iens in range(2):
            inputs[iens] = {**coeffs_trans[iens], **script_trans[iens]}
        config.update(
            {
                "inputs": inputs,
                "outputs": output_transmitters(config),
            }
        )

        fixed_port = range(unused_tcp_port, unused_tcp_port)
        custom_range = range(1024, 65535)

        service_config = EvaluatorServerConfig(custom_port_range=fixed_port)
        config["dispatch_uri"] = service_config.dispatch_uri
        ensemble = PrefectEnsemble(config, custom_port_range=custom_range)
        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")

        with evaluator.run() as mon:
            for event in mon.track():
                if isinstance(event.data, dict) and event.data.get("status") in [
                    "Failed",
                    "Stopped",
                ]:
                    mon.signal_done()

        assert evaluator._ensemble.get_status() == "Stopped"
        successful_realizations = evaluator._ensemble.get_successful_realizations()
        assert successful_realizations == config["realizations"]


@pytest.mark.timeout(60)
def test_run_prefect_for_function_defined_outside_py_environment(
    unused_tcp_port, coefficients, tmpdir, function_config
):
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

        config = function_config
        coeffs_trans = coefficient_transmitters(
            coefficients, config.get(ids.STORAGE)["storage_path"]
        )

        config["realizations"] = len(coefficients)
        config["executor"] = "local"
        config["steps"][0]["jobs"][0]["executable"] = pickle_func
        config["inputs"] = {iens: coeffs_trans[iens] for iens in range(2)}
        config["outputs"] = output_transmitters(config)

        fixed_port = range(unused_tcp_port, unused_tcp_port)
        custom_range = range(1024, 65535)

        service_config = EvaluatorServerConfig(custom_port_range=fixed_port)
        config["dispatch_uri"] = service_config.dispatch_uri

        ensemble = PrefectEnsemble(config, custom_port_range=custom_range)
        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")
        with evaluator.run() as mon:
            for event in mon.track():
                if event["type"] == ids.EVTYPE_EE_TERMINATED:
                    results = pickle.loads(event.data)
                if isinstance(event.data, dict) and event.data.get("status") in [
                    "Failed",
                    "Stopped",
                ]:
                    mon.signal_done()
        assert evaluator._ensemble.get_status() == "Stopped"
        successful_realizations = evaluator._ensemble.get_successful_realizations()
        assert successful_realizations == config["realizations"]
        expected_results = [
            pickle.loads(pickle_func)(coeffs) for coeffs in coefficients
        ]
        transmitter_futures = [
            res["function_output"].load() for res in results.values()
        ]
        results = asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*transmitter_futures)
        )
        assert expected_results == [res.data for res in results]


@pytest.mark.timeout(60)
def test_run_prefect_ensemble_with_path(unused_tcp_port, coefficients, source_root):
    with tmp(source_root / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        config.update(
            {
                "config_path": os.getcwd(),
                "realizations": 2,
                "executor": "local",
            }
        )
        inputs = {}
        coeffs_trans = coefficient_transmitters(
            coefficients, config.get(ids.STORAGE)["storage_path"]
        )
        script_trans = script_transmitters(config)
        for iens in range(2):
            inputs[iens] = {**coeffs_trans[iens], **script_trans[iens]}
        config.update(
            {
                "inputs": inputs,
                "outputs": output_transmitters(config),
            }
        )

        fixed_port = range(unused_tcp_port, unused_tcp_port)
        custom_range = range(1024, 65535)

        service_config = EvaluatorServerConfig(custom_port_range=fixed_port)
        config["config_path"] = Path(config["config_path"])
        config["run_path"] = Path(config["run_path"])
        config["storage"]["storage_path"] = Path(config["storage"]["storage_path"])
        config["dispatch_uri"] = service_config.dispatch_uri

        ensemble = PrefectEnsemble(config, custom_port_range=custom_range)

        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")

        with evaluator.run() as mon:
            for event in mon.track():
                if isinstance(event.data, dict) and event.data.get("status") in [
                    "Failed",
                    "Stopped",
                ]:
                    mon.signal_done()

        assert evaluator._ensemble.get_status() == "Stopped"
        successful_realizations = evaluator._ensemble.get_successful_realizations()
        assert successful_realizations == config["realizations"]


@pytest.mark.timeout(60)
def test_cancel_run_prefect_ensemble(unused_tcp_port, coefficients, source_root):
    with tmp(source_root / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        config.update(
            {
                "config_path": os.getcwd(),
                "realizations": 2,
                "executor": "local",
            }
        )
        inputs = {}
        coeffs_trans = coefficient_transmitters(
            coefficients, config.get(ids.STORAGE)["storage_path"]
        )
        script_trans = script_transmitters(config)
        for iens in range(2):
            inputs[iens] = {**coeffs_trans[iens], **script_trans[iens]}
        config.update(
            {
                "inputs": inputs,
                "outputs": output_transmitters(config),
            }
        )

        fixed_port = range(unused_tcp_port, unused_tcp_port)
        custom_range = range(1024, 65535)

        service_config = EvaluatorServerConfig(custom_port_range=fixed_port)
        config["config_path"] = Path(config["config_path"])
        config["run_path"] = Path(config["run_path"])
        config["storage"]["storage_path"] = Path(config["storage"]["storage_path"])
        config["dispatch_uri"] = service_config.dispatch_uri

        ensemble = PrefectEnsemble(config, custom_port_range=custom_range)

        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="2")

        with evaluator.run() as mon:
            cancel = True
            for _ in mon.track():
                if cancel:
                    mon.signal_cancel()
                    cancel = False

        assert evaluator._ensemble.get_status() == "Cancelled"


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="On darwin patching is unreliable since processes may use 'spawn'.",
)
def test_run_prefect_ensemble_exception(unused_tcp_port, coefficients, source_root):
    with tmp(source_root / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        config.update(
            {
                "config_path": os.getcwd(),
                "realizations": 2,
                "executor": "local",
            }
        )
        inputs = {}
        coeffs_trans = coefficient_transmitters(
            coefficients, config.get(ids.STORAGE)["storage_path"]
        )
        script_trans = script_transmitters(config)
        for iens in range(2):
            inputs[iens] = {**coeffs_trans[iens], **script_trans[iens]}
        config.update(
            {
                "inputs": inputs,
                "outputs": output_transmitters(config),
            }
        )

        fixed_port = range(unused_tcp_port, unused_tcp_port)
        custom_range = range(1024, 65535)

        service_config = EvaluatorServerConfig(custom_port_range=fixed_port)
        config["config_path"] = Path(config["config_path"])
        config["run_path"] = Path(config["run_path"])
        config["storage"]["storage_path"] = Path(config["storage"]["storage_path"])
        config["dispatch_uri"] = service_config.dispatch_uri

        ensemble = PrefectEnsemble(config, custom_port_range=custom_range)
        ensemble.get_flow = dummy_get_flow

        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")
        with evaluator.run() as mon:
            for event in mon.track():
                if event.data is not None and event.data.get("status") in [
                    "Failed",
                    "Stopped",
                ]:
                    mon.signal_done()
        assert evaluator._ensemble.get_status() == "Failed"
