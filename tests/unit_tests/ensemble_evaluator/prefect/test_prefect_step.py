import functools
import os
import pickle
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict

import cloudpickle
import prefect
import pytest
from prefect import Flow

import ert
import ert.ensemble_evaluator as ee
from ert.async_utils import get_event_loop
from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator._builder import InputBuilder, OutputBuilder
from ert.ensemble_evaluator._builder._job import JobBuilder
from ert.ensemble_evaluator._builder._prefect import _on_task_failure


def get_step(step_name, inputs, outputs, jobs, type_="unix"):
    input_map: Dict[str, ert.data.RecordTransmitter] = {}
    output_map: Dict[str, ert.data.RecordTransmitter] = {}
    real_source = "/real/0"
    step_source = "/real/0/step/0"
    step_builder = (
        ee.StepBuilder()
        .set_parent_source(source=real_source)
        .set_id("0")
        .set_name(step_name)
        .set_type(type_)
    )
    for idx, (name, executable, args) in enumerate(jobs):
        step_builder.add_job(
            JobBuilder()
            .set_id(str(idx))
            .set_index(str(idx))
            .set_name(name)
            .set_executable(executable)
            .set_args(args)
            .set_parent_source(step_source)
        )
    for name, path, mime, factory in inputs:
        transformation = ert.data.ExecutableTransformation(
            location=Path(path), mime=mime
        )
        step_builder.add_input(
            InputBuilder()
            .set_name(name)
            .set_transformation(transformation)
            .set_transmitter_factory(factory)
        )
    for name, path, mime, factory in outputs:
        step_builder.add_output(
            OutputBuilder()
            .set_name(name)
            .set_transformation(
                ert.data.SerializationTransformation(location=path, mime=mime)
            )
            .set_transmitter_factory(factory)
        )

    for input_ in step_builder._inputs:
        input_map[input_._name] = input_.transmitter_factory()()

    for output in step_builder._outputs:
        output_map[output._name] = output.transmitter_factory()()

    return step_builder.build(), input_map, output_map


@pytest.fixture()
def step_test_script_transmitter(test_data_path, transmitter_factory, script_name):
    async def transform_output(transmitter, location):
        transformation = ert.data.ExecutableTransformation(location=location)
        record = await transformation.to_record()
        await transmitter.transmit_record(record)

    script_transmitter = transmitter_factory("script")
    get_event_loop().run_until_complete(
        transform_output(
            transmitter=script_transmitter,
            location=test_data_path / script_name,
        )
    )
    return script_transmitter


def prefect_flow_run(ws_monitor, step, input_map, output_map, run_dir, **kwargs):
    """Use prefect-flow to run task from step output using task_inputs"""
    with prefect.context(url=ws_monitor.url, token=None, cert=None):
        with Flow("testing") as flow:
            task = step.get_task(
                output_transmitters=output_map, ens_id="test_ens_id", **kwargs
            )
            result = task(inputs=input_map)
        with run_dir.as_cwd():
            flow_run = flow.run()
    # Stop the mock evaluator WS server
    messages = ws_monitor.join_and_get_messages()
    return result, flow_run, messages


def assert_prefect_flow_run(
    result,
    flow_run,
    expect=True,
    step_type="unix",
    step_output=None,
    expected_result=None,
):
    """Assert if statements associated a prefect-flow-run are valid"""
    output_name = "output" if step_type == "unix" else "function_output"
    task_result = flow_run.result[result]
    # Check for all parametrizations
    assert task_result.is_successful() == expect
    assert flow_run.is_successful() == expect
    # Check when success is True
    if expect:
        assert len(task_result.result) == 1
        expected_uri = step_output[output_name]._uri
        output_uri = task_result.result[output_name]._uri
        assert expected_uri == output_uri
        # If function-step: Check result
        if step_type == "function":
            transmitted_record = get_event_loop().run_until_complete(
                task_result.result[output_name].load()
            )
            transmitted_result = transmitted_record.data
            assert transmitted_result == expected_result
    else:
        # Check when success is False
        assert isinstance(task_result.result, Exception)
        assert (
            "unix_test_script.py: error: unrecognized arguments: bar"
            in task_result.message
        )


@pytest.mark.parametrize("script_name", [("unix_test_script.py")])
@pytest.mark.parametrize(
    "job_args, error_test", [(["vas"], False), (["foo", "bar"], True)]
)
def test_unix_task(
    mock_ws_monitor,
    step_test_script_transmitter,
    transmitter_factory,
    job_args,
    error_test,
    tmpdir,
):
    """Test unix-step and error. Create step, run prefect flow, assert results"""
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
        jobs=[("script", Path("unix_test_script.py"), job_args)],
        type_="unix",
    )
    result, flow_run, messages = prefect_flow_run(
        ws_monitor=mock_ws_monitor,
        step=step,
        input_map=input_map,
        output_map=output_map,
        run_dir=tmpdir,
    )

    if not error_test:
        assert [
            msg for msg in messages if ids.EVTYPE_FM_JOB_SUCCESS in msg
        ], "no job success event"
        assert [
            msg for msg in messages if ids.EVTYPE_FM_STEP_SUCCESS in msg
        ], "no step success event"

    assert_prefect_flow_run(
        result=result,
        flow_run=flow_run,
        step_output=output_map,
        step_type="unix",
        expect=not error_test,
    )


@pytest.fixture
def internal_function():
    return "internal"


@pytest.fixture(params=["internal_function", "external_sum_function"])
def function(request):
    return request.getfixturevalue(request.param)


def test_function_step(
    function,
    mock_ws_monitor,
    input_transmitter_factory,
    transmitter_factory,
    tmpdir,
):
    """Test both internal and external function"""
    test_values = {"a": 42, "b": 24, "c": 6}
    if function == "internal":

        def sum_function(coeffs):
            return {"function_output": [sum(coeffs.values())]}

        sumfun = cloudpickle.dumps(sum_function)
    else:
        sumfun = function
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
        jobs=[("test_function", sumfun, None)],
        type_="function",
    )
    result, flow_run, _ = prefect_flow_run(
        ws_monitor=mock_ws_monitor,
        step=step,
        input_map=input_map,
        output_map=output_map,
        run_dir=tmpdir,
    )
    assert_prefect_flow_run(
        result=result,
        flow_run=flow_run,
        step_output=output_map,
        step_type="function",
        expected_result=pickle.loads(sumfun)(test_values)["function_output"],
    )


@pytest.mark.parametrize("retries,nfails,expect", [(3, 0, True), (1, 1, False)])
@pytest.mark.parametrize("script_name", [("unix_test_retry_script.py")])
def test_on_task_failure(
    mock_ws_monitor,
    step_test_script_transmitter,
    retries,
    nfails,
    expect,
    tmpdir,
):
    """Test both job and task failure of prefect-flow-run"""
    with tmpdir.as_cwd():
        step, input_map, output_map = get_step(
            step_name="test_step",
            inputs=[
                (
                    "script",
                    Path("unix_test_retry_script.py"),
                    "application/x-python",
                    lambda _t=step_test_script_transmitter: _t,
                )
            ],
            outputs=[],
            jobs=[("script", Path("unix_test_retry_script.py"), [tmpdir])],
            type_="unix",
        )
        os.mkdir(tmpdir.join("run_dir"))
        result, flow_run, messages = prefect_flow_run(
            ws_monitor=mock_ws_monitor,
            step=step,
            input_map=input_map,
            output_map=output_map,
            max_retries=retries,
            retry_delay=timedelta(seconds=1),
            on_failure=functools.partial(_on_task_failure, ens_id="test_ens_id"),
            run_dir=tmpdir.join("run_dir"),
        )
    task_result = flow_run.result[result]
    assert task_result.is_successful() == expect
    assert flow_run.is_successful() == expect
    fail_job_messages = [msg for msg in messages if ids.EVTYPE_FM_JOB_FAILURE in msg]
    fail_step_messages = [msg for msg in messages if ids.EVTYPE_FM_STEP_FAILURE in msg]
    expected_job_failed_messages = 2
    assert expected_job_failed_messages == len(fail_job_messages)
    assert nfails == len(fail_step_messages)
