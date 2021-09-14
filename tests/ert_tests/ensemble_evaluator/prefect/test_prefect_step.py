import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Callable, Type

import cloudpickle
import prefect
import pytest
from prefect import Flow

import ert
from ert_utils import tmp
import ert_shared.ensemble_evaluator.ensemble.builder as ee
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.ensemble.prefect import PrefectEnsemble


def input_transmitter(data, transmitter: Type[ert.data.RecordTransmitter]):
    record = ert.data.NumericalRecord(data=data)
    asyncio.get_event_loop().run_until_complete(transmitter.transmit_record(record))
    return transmitter


def create_script_transmitter(name: str, location: Path, transmitter_factory):
    script_transmitter = transmitter_factory(name)
    asyncio.get_event_loop().run_until_complete(
        script_transmitter.transmit_file(location, mime="application/octet-stream")
    )
    return script_transmitter


def step_output_transmitters(
    step, transmitter_factory: Callable[[str], Type[ert.data.RecordTransmitter]]
):
    transmitters = {}
    for output in step.get_outputs():
        transmitters[output.get_name()] = transmitter_factory(output.get_name())

    return transmitters


def get_step(step_name, inputs, outputs, jobs, type_="unix"):
    real_source = "/real/0"
    step_source = "/real/0/step/0"
    step_builder = ee.create_step_builder()
    for idx, (name, executable, args) in enumerate(jobs):
        step_builder.add_job(
            ee.create_job_builder()
            .set_id(str(idx))
            .set_name(name)
            .set_executable(executable)
            .set_args(args)
            .set_parent_source(step_source)
        )
    step_builder.set_parent_source(source=real_source)
    step_builder.set_id("0")
    step_builder.set_name(step_name)
    for name, path, mime in inputs:
        step_builder.add_input(
            ee.create_file_io_builder()
            .set_name(name)
            .set_path(Path(path))
            .set_mime(mime)
            .set_transformation(ert.data.ExecutableRecordTransformation())
        )
    for name, path, mime in outputs:
        step_builder.add_output(
            ee.create_file_io_builder().set_name(name).set_path(path).set_mime(mime)
        )
    step_builder.set_type(type_)
    return step_builder.build()


@pytest.fixture()
def step_test_script_transmitter(test_data_path, transmitter_factory):
    return create_script_transmitter(
        "script",
        location=test_data_path / "unix_test_script.py",
        transmitter_factory=transmitter_factory,
    )


@pytest.fixture()
def step_test_retry_script_transmitter(test_data_path, transmitter_factory):
    return create_script_transmitter(
        "script",
        location=test_data_path / "unix_test_retry_script.py",
        transmitter_factory=transmitter_factory,
    )


def test_unix_task(
    mock_ws_monitor, step_test_script_transmitter, step_output_transmitters_factory
):

    step = get_step(
        step_name="test_step",
        inputs=[("script", Path("unix_test_script.py"), "application/x-python")],
        outputs=[("output", Path("output.out"), "application/json")],
        jobs=[("script", Path("unix_test_script.py"), ["vas"])],
        type_="unix",
    )

    with prefect.context(url=mock_ws_monitor.url, token=None, cert=None):
        output_trans = step_output_transmitters_factory(step)
        with Flow("testing") as flow:
            task = step.get_task(output_transmitters=output_trans, ee_id="test_ee_id")
            result = task(inputs={"script": step_test_script_transmitter})
        with tmp():
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    mock_ws_monitor.join()

    task_result = flow_run.result[result]
    assert task_result.is_successful()
    assert flow_run.is_successful()

    assert len(task_result.result) == 1
    expected_uri = output_trans["output"]._uri
    output_uri = task_result.result["output"]._uri
    assert expected_uri == output_uri


def test_function_step(
    mock_ws_monitor, input_transmitter_factory, step_output_transmitters_factory
):
    test_values = [42, 24, 6]
    inputs = {"values": input_transmitter_factory("values", test_values)}

    def sum_function(values):
        return [sum(values)]

    step = get_step(
        step_name="test_step",
        inputs=[("values", "NA", "text/whatever")],
        outputs=[("output", Path("output.out"), "application/json")],
        jobs=[("test_function", cloudpickle.dumps(sum_function), None)],
        type_="function",
    )

    with prefect.context(url=mock_ws_monitor.url, token=None, cert=None):
        output_trans = step_output_transmitters_factory(step)
        with Flow("testing") as flow:
            task = step.get_task(output_transmitters=output_trans, ee_id="test_ee_id")
            result = task(inputs=inputs)
        with tmp():
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    mock_ws_monitor.join()

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
    expected_result = sum_function(values=test_values)
    assert expected_result == transmitted_result


def test_function_step_for_function_defined_outside_py_environment(
    mock_ws_monitor,
    external_sum_function,
    input_transmitter_factory,
    step_output_transmitters_factory,
):

    test_values = {"a": 42, "b": 24, "c": 6}
    inputs = {"coeffs": input_transmitter_factory("coeffs", test_values)}
    expected_result = 72

    step = get_step(
        step_name="test_step",
        inputs=[("coeffs", "NA", "text/whatever")],
        outputs=[("output", Path("output.out"), "application/json")],
        jobs=[("test_function", external_sum_function, None)],
        type_="function",
    )

    with prefect.context(url=mock_ws_monitor.url, token=None, cert=None):
        output_trans = step_output_transmitters_factory(step)
        with Flow("testing") as flow:
            task = step.get_task(output_transmitters=output_trans, ee_id="test_ee_id")
            result = task(inputs=inputs)
        with tmp():
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    mock_ws_monitor.join()

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
    assert [expected_result] == transmitted_result


def test_unix_step_error(
    mock_ws_monitor, step_test_script_transmitter, transmitter_factory
):
    step = get_step(
        step_name="test_step",
        inputs=[("test_script", Path("unix_test_script.py"), "application/x-python")],
        outputs=[("output", Path("output.out"), "application/json")],
        jobs=[("test_script", Path("unix_test_script.py"), ["foo", "bar"])],
        type_="unix",
    )

    with prefect.context(url=mock_ws_monitor.url, token=None, cert=None):
        output_trans = step_output_transmitters(
            step, transmitter_factory=transmitter_factory
        )
        with Flow("testing") as flow:
            task = step.get_task(output_transmitters=output_trans, ee_id="test_ee_id")
            result = task(inputs={"test_script": step_test_script_transmitter})
        with tmp():
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    mock_ws_monitor.join()

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


def test_on_task_failure(
    mock_ws_monitor, step_test_retry_script_transmitter, transmitter_factory
):
    mock_ensemble = _MockedPrefectEnsemble()

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

        with prefect.context(url=mock_ws_monitor.url, token=None, cert=None):
            output_trans = step_output_transmitters(
                step, transmitter_factory=transmitter_factory
            )
            with Flow("testing") as flow:
                task = step.get_task(
                    output_transmitters=output_trans,
                    ee_id="test_ee_id",
                    max_retries=3,
                    retry_delay=timedelta(seconds=1),
                    on_failure=mock_ensemble._on_task_failure,
                )
                result = task(inputs={"script": step_test_retry_script_transmitter})
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    messages = mock_ws_monitor.join_and_get_messages()

    task_result = flow_run.result[result]
    assert task_result.is_successful()
    assert flow_run.is_successful()

    fail_job_messages = [msg for msg in messages if ids.EVTYPE_FM_JOB_FAILURE in msg]
    fail_step_messages = [msg for msg in messages if ids.EVTYPE_FM_STEP_FAILURE in msg]

    expected_job_failed_messages = 2
    expected_step_failed_messages = 0
    assert expected_job_failed_messages == len(fail_job_messages)
    assert expected_step_failed_messages == len(fail_step_messages)


def test_on_task_failure_fail_step(
    mock_ws_monitor, step_test_retry_script_transmitter, transmitter_factory
):
    mock_ensemble = _MockedPrefectEnsemble()
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

        with prefect.context(url=mock_ws_monitor.url, token=None, cert=None):
            output_trans = step_output_transmitters(
                step, transmitter_factory=transmitter_factory
            )
            with Flow("testing") as flow:
                task = step.get_task(
                    output_transmitters=output_trans,
                    ee_id="test_ee_id",
                    max_retries=1,
                    retry_delay=timedelta(seconds=1),
                    on_failure=mock_ensemble._on_task_failure,
                )
                result = task(inputs={"script": step_test_retry_script_transmitter})
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    messages = mock_ws_monitor.join_and_get_messages()

    task_result = flow_run.result[result]
    assert not task_result.is_successful()
    assert not flow_run.is_successful()

    fail_job_messages = [msg for msg in messages if ids.EVTYPE_FM_JOB_FAILURE in msg]
    fail_step_messages = [msg for msg in messages if ids.EVTYPE_FM_STEP_FAILURE in msg]

    expected_job_failed_messages = 2
    expected_step_failed_messages = 1
    assert expected_job_failed_messages == len(fail_job_messages)
    assert expected_step_failed_messages == len(fail_step_messages)
