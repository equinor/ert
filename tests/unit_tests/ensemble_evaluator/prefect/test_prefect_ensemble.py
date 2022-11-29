import asyncio
import concurrent
import pickle
import sys
import uuid
from itertools import permutations
from pathlib import Path
from typing import Set

import cloudpickle
import prefect
import pytest
import websockets
from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent

import ert.ensemble_evaluator as ee
from ert.async_utils import get_event_loop
from ert.ensemble_evaluator import identifiers
from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator._builder._unix_task import UnixTask
from ert.ensemble_evaluator._wait_for_evaluator import wait_for_evaluator
from ert.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert.ensemble_evaluator.monitor import Monitor


def test_get_flow(
    poly_ensemble_builder,
    sum_coeffs_step,
    zero_degree_step,
    first_degree_step,
    second_degree_step,
    ensemble_size,
):
    """Assert flow-graph always holds no matter the input-permutation"""
    for permuted_steps in permutations(
        [sum_coeffs_step, zero_degree_step, first_degree_step, second_degree_step]
    ):
        real_builder = ee.RealizationBuilder().active(True)
        for permuted_step in permuted_steps:
            real_builder.add_step(permuted_step)

        poly_ensemble_builder = poly_ensemble_builder.set_forward_model(real_builder)
        ensemble = poly_ensemble_builder.build()

        for iens in range(ensemble_size):
            with prefect.context():
                flow = ensemble.get_flow([iens])

            # Get the ordered tasks and retrieve their step state.
            flow_steps = [
                task.step for task in flow.sorted_tasks() if isinstance(task, UnixTask)
            ]
            assert len(flow_steps) == 4

            realization_steps = list(
                ensemble.reals[iens].get_steps_sorted_topologically()
            )

            # Testing realization steps
            for step_ordering in [realization_steps, flow_steps]:
                mapping = {step.name: idx for idx, step in enumerate(step_ordering)}
                assert mapping["second_degree"] < mapping["zero_degree"]
                assert mapping["zero_degree"] < mapping["add_coeffs"]
                assert mapping["first_degree"] < mapping["add_coeffs"]
                assert mapping["second_degree"] < mapping["add_coeffs"]


def wait_until_done(monitor, event):
    """Manually close monitor if event-data both exists and state indicates done"""
    if isinstance(event.data, dict) and event.data.get("status") in [
        state.ENSEMBLE_STATE_FAILED,
        state.ENSEMBLE_STATE_STOPPED,
    ]:
        monitor.signal_done()


@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_run_prefect_ensemble(evaluator_config, poly_ensemble, ensemble_size):
    """Test successful realizations from prefect-run equals ensemble-size"""
    server = EnsembleEvaluator(poly_ensemble, evaluator_config, 0)
    asyncio.create_task(server.evaluator_server())

    await poly_ensemble.evaluate_async(evaluator_config, experiment_id=None)
    await server.stop()

    assert poly_ensemble.status == state.ENSEMBLE_STATE_STOPPED
    successful_realizations = poly_ensemble.get_successful_realizations()
    assert successful_realizations == ensemble_size


@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_cancel_run_prefect_ensemble(evaluator_config, poly_ensemble):
    """Test cancellation of prefect-run"""
    server = EnsembleEvaluator(poly_ensemble, evaluator_config, 0)
    server_task = asyncio.create_task(server.evaluator_server())

    await wait_for_evaluator(
        base_url=evaluator_config.url,
        token=evaluator_config.token,
        cert=evaluator_config.cert,
    )
    evaluation_task = asyncio.create_task(
        poly_ensemble.evaluate_async(evaluator_config, experiment_id=None)
    )
    cancel_event = CloudEvent(
        {
            "type": identifiers.EVTYPE_EE_USER_CANCEL,
            "source": f"/ert/monitor/0",
            "id": str(uuid.uuid1()),
        }
    )

    async with websockets.connect(f"{evaluator_config.client_uri}") as websocket:
        await websocket.send(to_json(cancel_event))

    await evaluation_task
    await server.stop()
    await server_task

    assert poly_ensemble.status == state.ENSEMBLE_STATE_CANCELLED


# This function is used by test_run_prefect_ensemble_exception, but
# it cannot be defined as a local/inline function, as it needs to
# be picklable
def dummy_get_flow(*args, **kwargs):
    raise RuntimeError()


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="On darwin patching is unreliable since processes may use 'spawn'.",
)
@pytest.mark.asyncio
async def test_run_prefect_ensemble_exception(evaluator_config, poly_ensemble):
    """Test prefect on flow with runtime-error"""
    poly_ensemble.get_flow = dummy_get_flow
    server = EnsembleEvaluator(poly_ensemble, evaluator_config, 0)
    server_task = asyncio.create_task(server.evaluator_server())

    await poly_ensemble.evaluate_async(evaluator_config, experiment_id=None)
    await server.stop()
    await server_task

    assert poly_ensemble.status == state.ENSEMBLE_STATE_FAILED


def function_that_fails_once(coeffs):
    # all retries load pickled function --> difficult to avoid assuming filesystem
    # Assumes sum(coeffs) is unique per realization
    sum_coeffs = sum(coeffs.values())
    run_path = Path("ran_once" + str(sum_coeffs))  # Avoid data races
    if not run_path.exists():
        run_path.touch()
        raise RuntimeError("This is an expected ERROR")
    try:
        run_path.unlink()
    except FileNotFoundError:
        # some other real beat us to it
        pass
    return {"function_output": []}


@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def just_not_yet_test_prefect_retries(
    evaluator_config, function_ensemble_builder_factory, tmpdir, ensemble_size
):

    """Evaluator fails once through pickled-fail-function. Asserts fail and retries"""
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    pickle_func = cloudpickle.dumps(function_that_fails_once)
    cloudpickle.unregister_pickle_by_value(sys.modules[__name__])

    builder = function_ensemble_builder_factory(pickle_func)
    ensemble = builder.set_retry_delay(2).set_id("0").build()

    server = EnsembleEvaluator(ensemble, evaluator_config, 0)
    server_task = asyncio.create_task(server.evaluator_server())
    con_info = evaluator_config.get_connection_info()
    await wait_for_evaluator(
        base_url=con_info.url, token=con_info.token, cert=con_info.cert
    )

    def run_function(con_config):
        error_event_reals: Set[str] = set()
        with Monitor(con_config.get_connection_info()) as monitor:
            for event in monitor.track():
                # Capture the job error messages
                if event.data is not None and "This is an expected ERROR" in str(
                    event.data
                ):
                    error_event_reals.update(event.data["reals"].keys())
        return error_event_reals

    with tmpdir.as_cwd():

        evaluation_task = asyncio.create_task(
            ensemble.evaluate_async(evaluator_config, experiment_id=None)
        )
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = await loop.run_in_executor(
                executor, run_function, evaluator_config
            )
            error_event_reals = future.result()

        await evaluation_task
        await server.stop()
        await server_task

        assert ensemble.status == state.ENSEMBLE_STATE_STOPPED
        successful_realizations = ensemble.get_successful_realizations()
        assert successful_realizations == ensemble_size
        # Check we get only one job error message per realization
        assert len(error_event_reals) == ensemble_size
        assert "0" in error_event_reals
        assert "1" in error_event_reals


@pytest.mark.timeout(60)
def test_prefect_no_retries(
    evaluator_config, function_ensemble_builder_factory, tmpdir
):

    """Evaluator tries and fails once. Asserts if job and step fails"""
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    pickle_func = cloudpickle.dumps(function_that_fails_once)
    cloudpickle.unregister_pickle_by_value(sys.modules[__name__])

    ensemble = (
        function_ensemble_builder_factory(pickle_func)
        .set_retry_delay(1)
        .set_max_retries(0)
        .set_id("0")
        .build()
    )
    evaluator = EnsembleEvaluator(ensemble, evaluator_config, 0)
    with tmpdir.as_cwd():
        # Get events
        event_list = []
        with evaluator.run() as mon:
            for event in mon.track():
                event_list.append(event)
                wait_until_done(mon, event)
        # Find if job and step failed
        step_failed = False
        job_failed = False
        for real in ensemble.snapshot.reals.values():
            for step in real.steps.values():
                for job in step.jobs.values():
                    if job.status == state.JOB_STATE_FAILURE:
                        job_failed = True
                        assert job.error == "This is an expected ERROR"
                        if step.status == state.STEP_STATE_FAILURE:
                            step_failed = True
        assert ensemble.status == state.ENSEMBLE_STATE_FAILED
        assert job_failed, f"Events: {event_list}"
        assert step_failed, f"Events: {event_list}"


@pytest.mark.timeout(60)
def test_run_prefect_for_function_defined_outside_py_environment(
    evaluator_config,
    coefficients,
    function_ensemble_builder_factory,
    ensemble_size,
    external_sum_function,
):
    """Ensemble built from outside env. Assert state, realizations and result"""
    # Build ensemble and run on server
    ensemble = (
        function_ensemble_builder_factory(external_sum_function)
        .set_retry_delay(1)
        .set_max_retries(0)
        .set_id("0")
        .build()
    )
    evaluator = EnsembleEvaluator(ensemble, evaluator_config, 0)
    with evaluator.run() as mon:
        for event in mon.track():
            if event["type"] == ids.EVTYPE_EE_TERMINATED:
                results = pickle.loads(event.data)
            wait_until_done(mon, event)
    assert evaluator._ensemble.status == state.ENSEMBLE_STATE_STOPPED
    successful_realizations = evaluator._ensemble.get_successful_realizations()
    assert successful_realizations == ensemble_size
    expected_results = [
        pickle.loads(external_sum_function)(coeffs)["function_output"]
        for coeffs in coefficients
    ]
    transmitter_futures = [res["function_output"].load() for res in results.values()]
    results = get_event_loop().run_until_complete(asyncio.gather(*transmitter_futures))
    assert expected_results == [res.data for res in results]
