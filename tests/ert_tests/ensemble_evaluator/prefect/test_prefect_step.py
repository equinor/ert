import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Type, Dict

import cloudpickle
import prefect
import pytest
from prefect import Flow
from functools import partial
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


def get_step(step_name, inputs, outputs, jobs, type_="unix"):
    input_map: Dict[str, ert.data.RecordTransmitter] = {}
    output_map: Dict[str, ert.data.RecordTransmitter] = {}
    real_source = "/real/0"
    step_source = "/real/0/step/0"
    step_builder = (
        ee.create_step_builder()
        .set_parent_source(source=real_source)
        .set_id("0")
        .set_name(step_name)
        .set_type(type_)
    )
    for idx, (name, executable, args) in enumerate(jobs):
        step_builder.add_job(
            ee.create_job_builder()
            .set_id(str(idx))
            .set_name(name)
            .set_executable(executable)
            .set_args(args)
            .set_parent_source(step_source)
        )
    for name, path, mime, factory in inputs:
        step_builder.add_input(
            ee.create_file_io_builder()
            .set_name(name)
            .set_path(Path(path))
            .set_mime(mime)
            .set_transformation(ert.data.ExecutableRecordTransformation())
            .set_transmitter_factory(factory)
        )
    for name, path, mime, factory in outputs:
        step_builder.add_output(
            ee.create_file_io_builder()
            .set_name(name)
            .set_path(path)
            .set_mime(mime)
            .set_transmitter_factory(factory)
        )

    for input_ in step_builder._inputs:
        input_map[input_._name] = input_.transmitter_factory()()

    for output in step_builder._outputs:
        output_map[output._name] = output.transmitter_factory()()

    return step_builder.build(), input_map, output_map


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


def test_unix_task(mock_ws_monitor, step_test_script_transmitter, transmitter_factory):
    step, input_map, output_map = get_step(
        step_name="test_step",
        inputs=[
            (
                "script",
                Path("unix_test_script.py"),
                "application/x-python",
                lambda _t=step_test_script_transmitter: _t,
            )
        ],
        outputs=[
            (
                "output",
                Path("output.out"),
                "application/json",
                partial(transmitter_factory, "output"),
            )
        ],
        jobs=[("script", Path("unix_test_script.py"), ["vas"])],
        type_="unix",
    )

    with prefect.context(url=mock_ws_monitor.url, token=None, cert=None):
        with Flow("testing") as flow:
            task = step.get_task(output_transmitters=output_map, ee_id="test_ee_id")
            result = task(inputs=input_map)
        with tmp():
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    mock_ws_monitor.join()

    task_result = flow_run.result[result]
    assert task_result.is_successful()
    assert flow_run.is_successful()

    assert len(task_result.result) == 1
    expected_uri = output_map["output"]._uri
    output_uri = task_result.result["output"]._uri
    assert expected_uri == output_uri


def test_function_step(mock_ws_monitor, input_transmitter_factory, transmitter_factory):
    test_values = [42, 24, 6]

    def sum_function(values):
        return {"output": [sum(values)]}

    step, input_map, output_map = get_step(
        step_name="test_step",
        inputs=[
            (
                "values",
                "NA",
                "text/whatever",
                partial(input_transmitter_factory, "values", test_values),
            )
        ],
        outputs=[
            (
                "output",
                Path("output.out"),
                "application/json",
                partial(transmitter_factory, "output"),
            )
        ],
        jobs=[("test_function", cloudpickle.dumps(sum_function), None)],
        type_="function",
    )

    with prefect.context(url=mock_ws_monitor.url, token=None, cert=None):
        with Flow("testing") as flow:
            task = step.get_task(output_transmitters=output_map, ee_id="test_ee_id")
            result = task(inputs=input_map)
        with tmp():
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    mock_ws_monitor.join()

    task_result = flow_run.result[result]
    assert task_result.is_successful()
    assert flow_run.is_successful()

    assert len(task_result.result) == 1
    expected_uri = output_map["output"]._uri
    output_uri = task_result.result["output"]._uri
    assert expected_uri == output_uri
    transmitted_record = asyncio.get_event_loop().run_until_complete(
        task_result.result["output"].load()
    )
    transmitted_result = transmitted_record.data
    expected_result = sum_function(values=test_values)["output"]
    assert expected_result == transmitted_result


def test_function_step_for_function_defined_outside_py_environment(
    mock_ws_monitor,
    external_sum_function,
    input_transmitter_factory,
    transmitter_factory,
):

    test_values = {"a": 42, "b": 24, "c": 6}
    expected_result = 72

    step, input_map, output_map = get_step(
        step_name="test_step",
        inputs=[
            (
                "coeffs",
                "NA",
                "text/whatever",
                partial(input_transmitter_factory, "coeffs", test_values),
            )
        ],
        outputs=[
            (
                "function_output",
                Path("output.out"),
                "application/json",
                partial(transmitter_factory, "function_output"),
            )
        ],
        jobs=[("test_function", external_sum_function, None)],
        type_="function",
    )

    with prefect.context(url=mock_ws_monitor.url, token=None, cert=None):
        with Flow("testing") as flow:
            task = step.get_task(output_transmitters=output_map, ee_id="test_ee_id")
            result = task(inputs=input_map)
        with tmp():
            flow_run = flow.run()

    # Stop the mock evaluator WS server
    mock_ws_monitor.join()

    task_result = flow_run.result[result]
    assert task_result.is_successful()
    assert flow_run.is_successful()

    assert len(task_result.result) == 1
    expected_uri = output_map["function_output"]._uri
    output_uri = task_result.result["function_output"]._uri
    assert expected_uri == output_uri
    transmitted_record = asyncio.get_event_loop().run_until_complete(
        task_result.result["function_output"].load()
    )
    transmitted_result = transmitted_record.data
    assert [expected_result] == transmitted_result


def test_unix_step_error(
    mock_ws_monitor, step_test_script_transmitter, transmitter_factory
):
    step, input_map, output_map = get_step(
        step_name="test_step",
        inputs=[
            (
                "test_script",
                Path("unix_test_script.py"),
                "application/x-python",
                lambda _t=step_test_script_transmitter: _t,
            )
        ],
        outputs=[
            (
                "output",
                Path("output.out"),
                "application/json",
                partial(transmitter_factory, "output"),
            )
        ],
        jobs=[("test_script", Path("unix_test_script.py"), ["foo", "bar"])],
        type_="unix",
    )

    with prefect.context(url=mock_ws_monitor.url, token=None, cert=None):
        with Flow("testing") as flow:
            task = step.get_task(output_transmitters=output_map, ee_id="test_ee_id")
            result = task(inputs=input_map)
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


def test_on_task_failure(mock_ws_monitor, step_test_retry_script_transmitter):
    mock_ensemble = _MockedPrefectEnsemble()

    with tmp() as runpath:
        step, input_map, output_map = get_step(
            step_name="test_step",
            inputs=[
                (
                    "script",
                    Path("unix_test_retry_script.py"),
                    "application/x-python",
                    lambda _t=step_test_retry_script_transmitter: _t,
                )
            ],
            outputs=[],
            jobs=[("script", Path("unix_test_retry_script.py"), [runpath])],
            type_="unix",
        )

        with prefect.context(url=mock_ws_monitor.url, token=None, cert=None):
            with Flow("testing") as flow:
                task = step.get_task(
                    output_transmitters=output_map,
                    ee_id="test_ee_id",
                    max_retries=3,
                    retry_delay=timedelta(seconds=1),
                    on_failure=mock_ensemble._on_task_failure,
                )
                result = task(inputs=input_map)
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


def test_on_task_failure_fail_step(mock_ws_monitor, step_test_retry_script_transmitter):
    mock_ensemble = _MockedPrefectEnsemble()
    with tmp() as runpath:
        step, input_map, output_map = get_step(
            step_name="test_step",
            inputs=[
                (
                    "script",
                    Path("unix_test_retry_script.py"),
                    "application/x-python",
                    lambda _t=step_test_retry_script_transmitter: _t,
                )
            ],
            outputs=[],
            jobs=[("script", Path("unix_test_retry_script.py"), [runpath])],
            type_="unix",
        )

        with prefect.context(url=mock_ws_monitor.url, token=None, cert=None):
            with Flow("testing") as flow:
                task = step.get_task(
                    output_transmitters=output_map,
                    ee_id="test_ee_id",
                    max_retries=1,
                    retry_delay=timedelta(seconds=1),
                    on_failure=mock_ensemble._on_task_failure,
                )
                result = task(inputs=input_map)
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
