import asyncio
import copy
import os
import os.path
import sys
import threading
from collections import defaultdict
from datetime import timedelta
from functools import partial
from itertools import permutations
from pathlib import Path

import ert3
import ert_shared.ensemble_evaluator.entity.ensemble as ee
import pytest
import yaml
from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.entity.unix_step import UnixTask
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.prefect_ensemble import PrefectEnsemble
from prefect import Flow
from prefect.run_configs import LocalRun
from tests.ensemble_evaluator.conftest import _mock_ws
from tests.utils import SOURCE_DIR, tmp


def parse_config(path):
    conf_path = Path(path).resolve()
    with open(conf_path, "r") as f:
        config = yaml.safe_load(f)
    for stage in config.get(ids.STAGES):
        for step in stage.get(ids.STEPS):
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
    transmitter = ert3.data.SharedDiskRecordTransmitter(
        name=name, storage_path=Path(storage_path)
    )
    asyncio.get_event_loop().run_until_complete(transmitter.transmit_data(data))
    return {name: transmitter}


def coefficient_transmitters(coefficients, storage_path):
    transmitters = defaultdict(dict)
    record_name = "coeffs"
    for iens, values in enumerate(coefficients):
        transmitters[iens] = input_transmitter(record_name, values, storage_path)
    return dict(transmitters)


def script_transmitters(config):
    transmitters = defaultdict(dict)
    for stage in config.get(ids.STAGES):
        for step in stage.get(ids.STEPS):
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
    transmitter = ert3.data.SharedDiskRecordTransmitter(
        name=name, storage_path=Path(storage_path)
    )
    with open(location, "rb") as f:
        asyncio.get_event_loop().run_until_complete(
            transmitter.transmit_data([f.read()], mime="text/x-python")
        )

    return {name: transmitter}


def output_transmitters(config):
    tmp_input_folder = "output_files"
    os.makedirs(tmp_input_folder)
    transmitters = defaultdict(dict)
    for stage in config.get(ids.STAGES):
        for step in stage.get(ids.STEPS):
            for output in step.get(ids.OUTPUTS):
                for iens in range(config.get(ids.REALIZATIONS)):
                    transmitters[iens][
                        output.get(ids.RECORD)
                    ] = ert3.data.SharedDiskRecordTransmitter(
                        output.get(ids.RECORD),
                        storage_path=Path(config.get(ids.STORAGE)["storage_path"]),
                    )
    return dict(transmitters)


def step_output_transmitters(step, storage_path):
    transmitters = {}
    for output in step.get_outputs():
        transmitters[output.get_name()] = ert3.data.SharedDiskRecordTransmitter(
            name=output.get_name(), storage_path=Path(storage_path)
        )

    return transmitters


@pytest.fixture()
def coefficients():
    return [{"a": a, "b": b, "c": c} for (a, b, c) in [(1, 2, 3), (4, 2, 1)]]


def get_step(step_name, inputs, outputs, jobs, url, type_="unix"):
    step_source = "/ert/ee/test_ee_id/real/0/stage/0/step/0"
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
    step_builder.set_ee_url(url)
    step_builder.set_source(source=step_source)
    step_builder.set_id(0)
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


def test_get_flow(coefficients, unused_tcp_port):
    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
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
        server_config = EvaluatorServerConfig(unused_tcp_port)
        for permuted_stages in permutations(config["stages"]):
            for stage_idx, stage in enumerate(permuted_stages):
                for permuted_steps in permutations(stage["steps"]):
                    permuted_config = copy.deepcopy(config)
                    permuted_config["stages"] = copy.deepcopy(permuted_stages)
                    permuted_config["stages"][stage_idx]["steps"] = permuted_steps
                    permuted_config["dispatch_uri"] = server_config.dispatch_uri
                    ensemble = PrefectEnsemble(permuted_config)

                    for iens in range(2):
                        flow = ensemble.get_flow(
                            ensemble._ee_id,
                            [iens],
                        )

                        # Get the ordered tasks and retrieve their stage and step ids.
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
                                step._name: idx
                                for idx, step in enumerate(step_ordering)
                            }
                            assert mapping["second_degree"] < mapping["zero_degree"]
                            assert mapping["zero_degree"] < mapping["add_coeffs"]
                            assert mapping["first_degree"] < mapping["add_coeffs"]
                            assert mapping["second_degree"] < mapping["add_coeffs"]


def test_unix_task(unused_tcp_port, tmpdir):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    script_location = (
        Path(SOURCE_DIR) / "test-data/local/prefect_test_case/unix_test_script.py"
    )
    input_ = script_transmitter("script", script_location, storage_path=tmpdir)
    step = get_step(
        step_name="test_step",
        inputs=[("script", Path("unix_test_script.py"), None)],
        outputs=[("output", Path("output.out"), None)],
        jobs=[("script", Path("unix_test_script.py"), ["vas"])],
        url=url,
        type_="unix",
    )

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
        inputs=[("values", "NA", None)],
        outputs=[("output", Path("output.out"), None)],
        jobs=[("test_function", sum_function, None)],
        url=url,
        type_="function",
    )

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


def test_unix_step_error(unused_tcp_port, tmpdir):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    script_location = (
        Path(SOURCE_DIR) / "test-data/local/prefect_test_case/unix_test_script.py"
    )
    input_ = script_transmitter("test_script", script_location, storage_path=tmpdir)
    step = get_step(
        step_name="test_step",
        inputs=[("test_script", Path("unix_test_script.py"), None)],
        outputs=[("output", Path("output.out"), None)],
        jobs=[("test_script", Path("unix_test_script.py"), ["foo", "bar"])],
        url=url,
        type_="unix",
    )

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


def test_on_task_failure(unused_tcp_port, tmpdir):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()
    script_location = (
        Path(SOURCE_DIR) / "test-data/local/prefect_test_case/unix_test_retry_script.py"
    )
    input_ = script_transmitter("script", script_location, storage_path=tmpdir)
    with tmp() as runpath:
        step = get_step(
            step_name="test_step",
            inputs=[("script", Path("unix_test_retry_script.py"), None)],
            outputs=[],
            jobs=[("script", Path("unix_test_retry_script.py"), [runpath])],
            url=url,
            type_="unix",
        )

        output_trans = step_output_transmitters(step, storage_path=tmpdir)
        with Flow("testing") as flow:
            task = step.get_task(
                output_transmitters=output_trans,
                ee_id="test_ee_id",
                max_retries=3,
                retry_delay=timedelta(seconds=1),
                on_failure=partial(PrefectEnsemble._on_task_failure, url=url),
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


def dummy_get_flow(*args, **kwargs):
    raise RuntimeError()


@pytest.mark.timeout(60)
def test_run_prefect_ensemble(unused_tcp_port, coefficients):
    test_path = Path(SOURCE_DIR) / "test-data/local/prefect_test_case"
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

        service_config = EvaluatorServerConfig(unused_tcp_port)
        config["dispatch_uri"] = service_config.dispatch_uri
        ensemble = PrefectEnsemble(config)
        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")

        with evaluator.run() as mon:
            for event in mon.track():
                if event.data is not None and event.data.get("status") in [
                    "Failed",
                    "Stopped",
                ]:
                    mon.signal_done()

        assert evaluator._snapshot.get_status() == "Stopped"
        successful_realizations = evaluator._snapshot.get_successful_realizations()
        assert successful_realizations == config["realizations"]


@pytest.mark.timeout(60)
def test_run_prefect_ensemble_with_path(unused_tcp_port, coefficients):
    with tmp(os.path.join(SOURCE_DIR, "test-data/local/prefect_test_case")):
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

        service_config = EvaluatorServerConfig(unused_tcp_port)
        config["config_path"] = Path(config["config_path"])
        config["run_path"] = Path(config["run_path"])
        config["storage"]["storage_path"] = Path(config["storage"]["storage_path"])
        config["dispatch_uri"] = service_config.dispatch_uri

        ensemble = PrefectEnsemble(config)

        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")

        with evaluator.run() as mon:
            for event in mon.track():
                if event.data is not None and event.data.get("status") in [
                    "Failed",
                    "Stopped",
                ]:
                    mon.signal_done()

        assert evaluator._snapshot.get_status() == "Stopped"
        successful_realizations = evaluator._snapshot.get_successful_realizations()
        assert successful_realizations == config["realizations"]


@pytest.mark.timeout(60)
def test_cancel_run_prefect_ensemble(unused_tcp_port, coefficients):
    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
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

        service_config = EvaluatorServerConfig(unused_tcp_port)
        config["config_path"] = Path(config["config_path"])
        config["run_path"] = Path(config["run_path"])
        config["storage"]["storage_path"] = Path(config["storage"]["storage_path"])
        config["dispatch_uri"] = service_config.dispatch_uri

        ensemble = PrefectEnsemble(config)

        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="2")

        with evaluator.run() as mon:
            cancel = True
            for _ in mon.track():
                if cancel:
                    mon.signal_cancel()
                    cancel = False

        assert evaluator._snapshot.get_status() == "Cancelled"


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="On darwin patching is unreliable since processes may use 'spawn'.",
)
def test_run_prefect_ensemble_exception(unused_tcp_port, coefficients):
    with tmp(os.path.join(SOURCE_DIR, "test-data/local/prefect_test_case")):
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

        service_config = EvaluatorServerConfig(unused_tcp_port)
        config["config_path"] = Path(config["config_path"])
        config["run_path"] = Path(config["run_path"])
        config["storage"]["storage_path"] = Path(config["storage"]["storage_path"])
        config["dispatch_uri"] = service_config.dispatch_uri

        ensemble = PrefectEnsemble(config)
        ensemble.get_flow = dummy_get_flow

        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")
        with evaluator.run() as mon:
            for event in mon.track():
                if event.data is not None and event.data.get("status") in [
                    "Failed",
                    "Stopped",
                ]:
                    mon.signal_done()
        assert evaluator._snapshot.get_status() == "Failed"
