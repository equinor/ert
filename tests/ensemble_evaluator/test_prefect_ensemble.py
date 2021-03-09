from pathlib import Path
import sys
import os
import os.path
import yaml
import pytest
import threading
import json
import copy
from datetime import timedelta
from functools import partial
from itertools import permutations
from prefect import Flow
from tests.utils import SOURCE_DIR, tmp
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.prefect_ensemble.prefect_ensemble import (
    PrefectEnsemble,
)
from ert_shared.ensemble_evaluator.prefect_ensemble.unix_step import UnixStep
from ert_shared.ensemble_evaluator.prefect_ensemble.function_step import FunctionStep
from ert_shared.ensemble_evaluator.prefect_ensemble.storage_driver import (
    storage_driver_factory,
)
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.prefect_ensemble.client import Client
from tests.ensemble_evaluator.conftest import _mock_ws


def parse_config(path):
    conf_path = Path(path).resolve()
    with open(conf_path, "r") as f:
        return yaml.safe_load(f)


def input_files(config, coefficients):
    paths = {}
    storage = storage_driver_factory(config.get(ids.STORAGE), ".")
    file_name = "coeffs.json"
    for iens, values in enumerate(coefficients):
        with open(file_name, "w") as f:
            json.dump(values, f)
        paths[iens] = (storage.store(file_name, iens),)
    return paths


@pytest.fixture()
def coefficients():
    return [{"a": a, "b": b, "c": c} for (a, b, c) in [(1, 2, 3), (4, 2, 1)]]


def test_get_id():
    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        ensemble = PrefectEnsemble(config)

        all_ids = set(
            [
                (
                    realization.get_iens(),
                    stage.get_id(),
                    step.get_id(),
                    job.get_id(),
                )
                for realization in ensemble.get_reals()
                for stage in realization.get_stages()
                for step in stage.get_steps()
                for job in step.get_jobs()
            ]
        )

        ids_from_get_id = set()
        for real_idx in range(ensemble.config[ids.REALIZATIONS]):
            real_id = ensemble.get_reals()[real_idx].get_iens()
            for stage in config[ids.STAGES]:
                stage_id = ensemble.get_id(
                    real_id,
                    stage["name"],
                )
                for step in stage[ids.STEPS]:
                    step_id = ensemble.get_id(
                        real_id,
                        stage["name"],
                        step_name=step["name"],
                    )
                    for job_idx in range(len(step[ids.JOBS])):
                        job_id = ensemble.get_id(
                            real_id,
                            stage["name"],
                            step_name=step["name"],
                            job_index=job_idx,
                        )
                        ids_from_get_id.add((real_id, stage_id, step_id, job_id))

        assert ids_from_get_id == all_ids


def test_get_ordering():
    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        for permuted_stages in permutations(config["stages"]):
            for stage_idx, stage in enumerate(permuted_stages):
                for permuted_steps in permutations(stage["steps"]):
                    permuted_config = copy.deepcopy(config)
                    permuted_config["stages"] = copy.deepcopy(permuted_stages)
                    permuted_config["stages"][stage_idx]["steps"] = permuted_steps

                    ensemble = PrefectEnsemble(permuted_config)

                    stages_steps = [
                        (item["stage_name"], item["name"])
                        for item in ensemble.get_ordering(0)
                    ]

                    assert stages_steps.index(
                        ("calculate_coeffs", "second_degree")
                    ) < stages_steps.index(("calculate_coeffs", "zero_degree"))
                    assert stages_steps.index(
                        ("calculate_coeffs", "zero_degree")
                    ) < stages_steps.index(("sum_coeffs", "add_coeffs"))
                    assert stages_steps.index(
                        ("calculate_coeffs", "first_degree")
                    ) < stages_steps.index(("sum_coeffs", "add_coeffs"))
                    assert stages_steps.index(
                        ("calculate_coeffs", "second_degree")
                    ) < stages_steps.index(("sum_coeffs", "add_coeffs"))


def test_get_ordering_exception():
    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        # Introduce a circular dependency.
        config["stages"][1]["steps"][2]["inputs"] = ["poly_0.out"]
        ensemble = PrefectEnsemble(config)
        with pytest.raises(ValueError, match="Could not reorder workflow"):
            ensemble.get_ordering(0)


def test_get_flow(coefficients):
    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        config.update(
            {
                "config_path": os.getcwd(),
                ids.REALIZATIONS: 2,
                ids.EXECUTOR: "local",
                "input_files": input_files(config, coefficients),
            }
        )

        def _get_ids(ensemble, iens, stage_name, step_name):
            stage_id = ensemble.get_id(iens, stage_name)
            step_id = ensemble.get_id(iens, stage_name, step_name=step_name)
            return stage_id, step_id

        for permuted_stages in permutations(config["stages"]):
            for stage_idx, stage in enumerate(permuted_stages):
                for permuted_steps in permutations(stage["steps"]):
                    permuted_config = copy.deepcopy(config)
                    permuted_config["stages"] = copy.deepcopy(permuted_stages)
                    permuted_config["stages"][stage_idx]["steps"] = permuted_steps

                    ensemble = PrefectEnsemble(permuted_config)

                    flow = ensemble.get_flow(
                        ensemble._ee_id,
                        ensemble._ee_dispach_url,
                        ensemble.config["input_files"],
                        [0, 1],
                    )

                    # Get the ordered tasks and retrieve their stage and step ids.
                    task_ids = [
                        (task.get_stage_id(), task.get_step_id())
                        for task in flow.sorted_tasks()
                        if task.name == "UnixStep"
                    ]
                    assert len(task_ids) == 8

                    # Check some task dependencies.
                    for iens in range(2):
                        assert task_ids.index(
                            _get_ids(
                                ensemble, iens, "calculate_coeffs", "second_degree"
                            )
                        ) < task_ids.index(
                            _get_ids(ensemble, iens, "calculate_coeffs", "zero_degree")
                        )
                        assert task_ids.index(
                            _get_ids(ensemble, iens, "calculate_coeffs", "zero_degree")
                        ) < task_ids.index(
                            _get_ids(ensemble, iens, "sum_coeffs", "add_coeffs")
                        )
                        assert task_ids.index(
                            _get_ids(ensemble, iens, "calculate_coeffs", "first_degree")
                        ) < task_ids.index(
                            _get_ids(ensemble, iens, "sum_coeffs", "add_coeffs")
                        )
                        assert task_ids.index(
                            _get_ids(
                                ensemble, iens, "calculate_coeffs", "second_degree"
                            )
                        ) < task_ids.index(
                            _get_ids(ensemble, iens, "sum_coeffs", "add_coeffs")
                        )


def test_unix_step(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    def _on_task_failure(task, state):
        raise Exception(state.message)

    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        storage = storage_driver_factory(config=config.get(ids.STORAGE), run_path=".")
        resource = storage.store("unix_test_script.py")
        jobs = [
            {
                ids.ID: "0",
                ids.NAME: "test_script",
                ids.EXECUTABLE: "unix_test_script.py",
                ids.ARGS: ["vas"],
            }
        ]
        step = {
            ids.OUTPUTS: ["output.out"],
            ids.IENS: 1,
            ids.STEP_ID: "step_id_0",
            ids.STAGE_ID: "stage_id_0",
            ids.JOBS: jobs,
        }

        stage_task = UnixStep(
            step=step,
            resources=[resource],
            cmd="python3",
            url=url,
            ee_id="ee_id_0",
            on_failure=_on_task_failure,
            run_path=config.get(ids.RUN_PATH),
            storage_config=config.get(ids.STORAGE),
            max_retries=1,
            retry_delay=timedelta(seconds=2),
        )

        flow = Flow("testing")
        flow.add_task(stage_task)
        flow_run = flow.run()

        # Stop the mock evaluator WS server
        with Client(url) as c:
            c.send("stop")
        mock_ws_thread.join()

        task_result = flow_run.result[stage_task]
        assert task_result.is_successful()
        assert flow_run.is_successful()

        assert len(task_result.result["outputs"]) == 1
        expected_path = storage.get_storage_path(1) / "output.out"
        output_path = flow_run.result[stage_task].result["outputs"][0]
        assert expected_path == output_path
        assert output_path.exists()


def test_function_step(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    def _on_task_failure(task, state):
        raise Exception(state.message)

    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        storage = storage_driver_factory(config=config.get(ids.STORAGE), run_path=".")

        def sum_function(values):
            return sum(values)

        jobs = [
            {
                ids.ID: "0",
                ids.NAME: "test_script",
                ids.EXECUTABLE: sum_function,
                "output": "output.out",
            }
        ]
        test_values = {"values": [42, 24, 6]}
        step = {
            ids.JOBS: jobs,
            ids.STEP_ID: "step_id_0",
            ids.STAGE_ID: "stage_id_0",
            ids.IENS: 1,
            "step_input": test_values,
        }

        function_task = FunctionStep(
            step=step,
            url=url,
            ee_id="ee_id_0",
            on_failure=_on_task_failure,
            storage_config=config.get(ids.STORAGE),
            max_retries=1,
            retry_delay=timedelta(seconds=2),
        )

        flow = Flow("testing")
        flow.add_task(function_task)
        flow_run = flow.run()

        # Stop the mock evaluator WS server
        with Client(url) as c:
            c.send("stop")
        mock_ws_thread.join()

        task_result = flow_run.result[function_task]
        assert task_result.is_successful()
        assert flow_run.is_successful()

        assert len(task_result.result["outputs"]) == 1
        expected_path = storage.get_storage_path(1) / "output.out"
        output_path = flow_run.result[function_task].result["outputs"][0]
        assert expected_path == output_path
        assert output_path.exists()
        with open(output_path, "r") as f:
            result = json.load(f)
        assert sum_function(**test_values) == result


def test_unix_step_error(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    def _on_task_failure(task, state):
        raise Exception(state.message)

    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case", False):
        config = parse_config("config.yml")
        storage = storage_driver_factory(config=config.get(ids.STORAGE), run_path=".")
        resource = storage.store("unix_test_script.py")
        jobs = [
            {
                ids.ID: "0",
                ids.NAME: "test_script",
                ids.EXECUTABLE: "unix_test_script.py",
                ids.ARGS: ["foo", "bar"],
            }
        ]
        step = {
            ids.OUTPUTS: ["output.out"],
            ids.IENS: 1,
            ids.STEP_ID: "step_id_0",
            ids.STAGE_ID: "stage_id_0",
            ids.JOBS: jobs,
        }

        stage_task = UnixStep(
            step=step,
            resources=[resource],
            cmd="python3",
            url=url,
            ee_id="ee_id_0",
            on_failure=_on_task_failure,
            run_path=config.get(ids.RUN_PATH),
            storage_config=config.get(ids.STORAGE),
            max_retries=1,
            retry_delay=timedelta(seconds=2),
        )

        flow = Flow("testing")
        flow.add_task(stage_task)
        flow_run = flow.run()

        # Stop the mock evaluator WS server
        with Client(url) as c:
            c.send("stop")
        mock_ws_thread.join()

        task_result = flow_run.result[stage_task]
        assert not task_result.is_successful()
        assert not flow_run.is_successful()

        assert isinstance(task_result.result, Exception)
        assert (
            "Script test_script failed with exception usage: unix_test_script.py [-h] argument"
            in task_result.message
        )


def test_on_task_failure(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    messages = []
    mock_ws_thread = threading.Thread(
        target=partial(_mock_ws, messages=messages), args=(host, unused_tcp_port)
    )

    mock_ws_thread.start()

    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case", False):
        config = parse_config("config.yml")
        storage = storage_driver_factory(config=config.get(ids.STORAGE), run_path=".")
        resource = storage.store("unix_test_retry_script.py")
        jobs = [
            {
                ids.ID: "0",
                ids.NAME: "test_script",
                ids.EXECUTABLE: "unix_test_retry_script.py",
                ids.ARGS: [],
            }
        ]
        step = {
            ids.OUTPUTS: [],
            ids.IENS: 1,
            ids.STEP_ID: "step_id_0",
            ids.STAGE_ID: "stage_id_0",
            ids.JOBS: jobs,
        }

        stage_task = UnixStep(
            step=step,
            resources=[resource],
            cmd="python3",
            url=url,
            ee_id="ee_id_0",
            on_failure=partial(PrefectEnsemble._on_task_failure, url=url),
            run_path=config.get(ids.RUN_PATH),
            storage_config=config.get(ids.STORAGE),
            max_retries=3,
            retry_delay=timedelta(seconds=1),
        )

        flow = Flow("testing")
        flow.add_task(stage_task)
        flow_run = flow.run()

        # Stop the mock evaluator WS server
        with Client(url) as c:
            c.send("stop")
        mock_ws_thread.join()

        task_result = flow_run.result[stage_task]
        assert task_result.is_successful()
        assert flow_run.is_successful()

        fail_job_messages = [
            msg for msg in messages if ids.EVTYPE_FM_JOB_FAILURE in msg
        ]
        fail_step_messages = [
            msg for msg in messages if ids.EVTYPE_FM_STEP_FAILURE in msg
        ]

        expected_job_failed_messages = 2
        expected_step_failed_messages = 0
        assert expected_job_failed_messages == len(fail_job_messages)
        assert expected_step_failed_messages == len(fail_step_messages)


def dummy_get_flow(*args, **kwargs):
    raise RuntimeError()


@pytest.mark.timeout(60)
def test_run_prefect_ensemble(unused_tcp_port, coefficients):
    with tmp(Path(SOURCE_DIR) / "test-data/local/prefect_test_case"):
        config = parse_config("config.yml")
        config.update(
            {
                "config_path": os.getcwd(),
                "realizations": 2,
                "executor": "local",
                "input_files": input_files(config, coefficients),
            }
        )

        service_config = EvaluatorServerConfig(unused_tcp_port)
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
                "config_path": Path.cwd(),
                "realizations": 2,
                "executor": "local",
                "input_files": input_files(config, coefficients),
            }
        )

        config["config_path"] = Path(config["config_path"])
        config["run_path"] = Path(config["run_path"])
        config["storage"]["storage_path"] = Path(config["storage"]["storage_path"])

        service_config = EvaluatorServerConfig(unused_tcp_port)
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
                "input_files": input_files(config, coefficients),
            }
        )

        service_config = EvaluatorServerConfig(unused_tcp_port)
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
                ids.REALIZATIONS: 2,
                ids.EXECUTOR: "local",
                "input_files": input_files(config, coefficients),
            }
        )

        service_config = EvaluatorServerConfig(unused_tcp_port)

        ensemble = PrefectEnsemble(config)
        evaluator = EnsembleEvaluator(ensemble, service_config, 0, ee_id="1")

        ensemble.get_flow = dummy_get_flow
        with evaluator.run() as mon:
            for event in mon.track():
                if event.data is not None and event.data.get("status") in [
                    "Failed",
                    "Stopped",
                ]:
                    mon.signal_done()
        assert evaluator._snapshot.get_status() == "Failed"
