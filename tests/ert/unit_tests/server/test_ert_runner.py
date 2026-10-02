import json
from collections.abc import Iterator
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient

from ert.config import ErtConfig, QueueSystem
from ert.config.gen_kw_config import GenKwConfig
from ert.ensemble_evaluator import EndEvent, EvaluatorServerConfig
from ert.run_models import SingleTestRun
from ert.run_models.run_model import RunModel
from ert.server.endpoints import ert_runner
from ert.server.endpoints.experiment_runs import (
    ExperimentRunnerState,
    _experiments,
    _get_experiment,
    verify_auth,
)
from ert.storage import LocalStorage, open_storage
from everest.everserver.server import ExperimentState, ExperimentStatus


def test_that_run_args_deserializes_selected_parameter_configurations() -> None:
    parameters = [
        GenKwConfig.model_validate(
            {
                "name": name,
                "distribution": {"name": "normal", "mean": 0, "std": 1},
                "update_strategy": None,
            }
        )
        for name in ["SELECTED_B", "SELECTED_A"]
    ]

    args = ert_runner.RunArgs.model_validate_json(
        json.dumps(
            {
                "mode": "es_mda",
                "experiment_name": "parameter-selection",
                "parameter_configuration": [
                    parameter.model_dump(mode="json") for parameter in parameters
                ],
            }
        )
    )

    assert args.parameter_configuration == parameters
    assert all(
        isinstance(parameter, GenKwConfig) for parameter in args.parameter_configuration
    )
    assert all(
        parameter.update_strategy is None for parameter in args.parameter_configuration
    )


def test_that_run_args_preserves_omitted_parameter_configuration() -> None:
    args = ert_runner.RunArgs.model_validate_json(
        '{"mode": "es_mda", "experiment_name": "parameter-selection"}'
    )

    assert not hasattr(args, "parameter_configuration")


def test_that_run_args_preserves_empty_parameter_configuration() -> None:
    args = ert_runner.RunArgs.model_validate_json(
        '{"mode": "es_mda", "experiment_name": "parameter-selection", '
        '"parameter_configuration": []}'
    )

    assert args.parameter_configuration == []


@pytest.mark.asyncio
async def test_that_start_uses_the_registered_model_and_config_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry: dict[str, ert_runner.ErtExperimentRunnerState] = {}
    monkeypatch.setattr(ert_runner, "_experiments", registry)
    config = Mock(
        spec=ErtConfig,
        config_path="config.ert",
        runpath_file=".ert_runpath_list",
        ens_path="storage",
    )
    request = ert_runner.StartErtExperimentRequest.model_construct(
        config=config,
        args=ert_runner.RunArgs(
            mode="ensemble_experiment", experiment_name="registration-test"
        ),
    )
    tasks = BackgroundTasks()
    model = Mock(spec=RunModel)
    writer = Mock()
    model._storage = writer
    create_model = Mock(return_value=model)
    monkeypatch.setattr(ert_runner, "create_model", create_model)

    registration = await ert_runner.register(request)
    writer.close.assert_not_called()
    assert model._storage is writer
    config_id = json.loads(bytes(registration.body))["config_id"]
    response = await ert_runner.start_experiment_ert(
        ert_runner._get_ert_experiment(config_id), tasks
    )

    assert response.status_code == 200
    assert response.body == b""
    state = registry[config_id]
    assert isinstance(state, ert_runner.ErtExperimentRunnerState)
    assert state.run_model is model
    assert state.config_path == "config.ert"
    assert state.run_path == ".ert_runpath_list"
    assert state.storage_path == "storage"
    assert state.start_time_unix is not None
    create_model.assert_called_once_with(config, request.args, state.status_queue)
    assert len(tasks.tasks) == 1
    assert tasks.tasks[0].func is ert_runner.run_ert
    assert tasks.tasks[0].args == (state,)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "failed", "exception"])
@pytest.mark.parametrize("rerun_failed_realizations", [False, True])
async def test_that_execution_retains_its_model_after_termination(
    monkeypatch: pytest.MonkeyPatch, outcome: str, rerun_failed_realizations: bool
) -> None:
    experiment_id = str(uuid4())
    model = Mock(spec=RunModel)
    state = ert_runner.ErtExperimentRunnerState(
        run_model=model, status_queue=SimpleQueue()
    )
    monkeypatch.setitem(_experiments, experiment_id, state)
    model.queue_config = SimpleNamespace(queue_system=QueueSystem.LOCAL)

    def execute(
        evaluator_config: EvaluatorServerConfig, *, rerun_failed_realizations: bool
    ) -> None:
        assert state.run_model is model
        if outcome == "exception":
            raise RuntimeError("Simulation raised")
        state.status_queue.put(EndEvent(failed=outcome == "failed", msg=outcome))

    model.start_simulations_thread.side_effect = execute
    create_model = Mock(side_effect=AssertionError("Model already registered"))
    monkeypatch.setattr(ert_runner, "create_model", create_model)
    await ert_runner.run_ert(state, rerun_failed_realizations=rerun_failed_realizations)

    assert model.start_simulations_thread.call_args.kwargs == {
        "rerun_failed_realizations": rerun_failed_realizations
    }
    create_model.assert_not_called()
    assert _get_experiment(experiment_id) is state
    assert state.run_model is model
    assert state.status.status == (
        ExperimentState.completed if outcome == "completed" else ExperimentState.failed
    )
    assert state.events[-1] == EndEvent(
        failed=outcome != "completed",
        msg="Simulation raised" if outcome == "exception" else outcome,
    )


@pytest.mark.asyncio
async def test_that_model_creation_error_does_not_register_a_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry: dict[str, ert_runner.ErtExperimentRunnerState] = {}
    monkeypatch.setattr(ert_runner, "_experiments", registry)
    monkeypatch.setattr(
        ert_runner, "create_model", Mock(side_effect=ValueError("Invalid config"))
    )

    request = ert_runner.StartErtExperimentRequest.model_construct(
        config=Mock(spec=ErtConfig),
        args=ert_runner.RunArgs(
            mode="ensemble_experiment", experiment_name="registration-test"
        ),
    )

    with pytest.raises(ValueError, match="Invalid config"):
        await ert_runner.register(request)
    assert not registry


@pytest.mark.asyncio
async def test_that_failed_registration_releases_partially_initialized_storage(
    minimum_case: ErtConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry: dict[str, ert_runner.ErtExperimentRunnerState] = {}
    monkeypatch.setattr(ert_runner, "_experiments", registry)
    monkeypatch.setattr(LocalStorage, "LOCK_TIMEOUT", 0)

    def fail_after_opening_storage(self: SingleTestRun, context: object) -> None:
        RunModel.model_post_init(self, context)
        raise RuntimeError("Invalid prior ensemble")

    monkeypatch.setattr(SingleTestRun, "model_post_init", fail_after_opening_storage)
    request = ert_runner.StartErtExperimentRequest(
        config=minimum_case,
        args=ert_runner.RunArgs.model_validate(
            {
                "mode": "test_run",
                "experiment_name": "registration-test",
                "current_ensemble": "prior",
                "active_realizations": None,
            }
        ),
    )

    with pytest.raises(RuntimeError, match="Invalid prior ensemble") as error:
        await ert_runner.register(request)

    assert error.value.__traceback__ is not None
    assert not registry
    with open_storage(minimum_case.ens_path, mode="w") as storage:
        assert storage.can_write


@pytest.mark.asyncio
async def test_that_cancellation_before_start_does_not_start_simulations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_id = str(uuid4())
    model = Mock(spec=RunModel)
    model._storage = Mock(spec=LocalStorage)
    state = ert_runner.ErtExperimentRunnerState(
        status=ExperimentStatus(status=ExperimentState.stopped, message="Cancelled"),
        run_model=model,
        status_queue=SimpleQueue(),
    )
    monkeypatch.setitem(_experiments, experiment_id, state)
    create_model = Mock()
    monkeypatch.setattr(ert_runner, "create_model", create_model)

    await ert_runner.run_ert(state)

    create_model.assert_not_called()
    model.start_simulations_thread.assert_not_called()
    model._storage.close.assert_called_once_with()
    assert state.events == [
        EndEvent(failed=True, msg="Experiment cancelled before start.")
    ]


@pytest.fixture
def runner_client() -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(ert_runner.router)
    app.dependency_overrides[verify_auth] = lambda: None
    app.dependency_overrides[ert_runner._with_runtime_plugins] = lambda: None
    app.dependency_overrides[ert_runner.get_storage] = lambda: None
    with TestClient(app) as client:
        yield client


def test_that_inspection_and_runpath_requests_reuse_the_registered_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner_client: TestClient
) -> None:
    runpath = tmp_path / "realization-0"
    runpath.mkdir()
    model = Mock(spec=RunModel)
    model.paths = [str(runpath)]
    model._storage = Mock()
    model.get_number_of_existing_runpaths.return_value = 1
    model.get_number_of_active_realizations.return_value = 2
    model.supports_rerunning_failed_realizations = True
    model._create_mask_from_failed_realizations.return_value = [False, True]
    state = ert_runner.ErtExperimentRunnerState(
        run_model=model, status_queue=SimpleQueue()
    )
    config_id = str(uuid4())
    monkeypatch.setitem(_experiments, config_id, state)
    create_model = Mock(side_effect=AssertionError("Model already registered"))
    monkeypatch.setattr(ert_runner, "create_model", create_model)
    payload = {"config_id": config_id}

    response = runner_client.post("/experiment_runs/runpath", params=payload)
    assert response.status_code == 200
    response = runner_client.post("/experiment_runs/runmodel", params=payload)
    assert response.status_code == 200
    assert response.json() == {
        "number_of_existing_runpaths": 1,
        "number_of_active_realizations": 2,
        "supports_rerunning_failed_realizations": True,
    }
    response = runner_client.request(
        "GET", "/experiment_runs/failed_realizations", params=payload
    )
    assert response.status_code == 200
    assert response.json() == {"failed_realizations": [False, True]}
    response = runner_client.request(
        "DELETE", "/experiment_runs/runpath", params=payload
    )
    assert response.status_code == 200
    assert not runpath.exists()
    response = runner_client.post("/experiment_runs/runpath", params=payload)
    assert response.status_code == 404
    model._storage.close.assert_not_called()
    create_model.assert_not_called()


@pytest.mark.parametrize(
    ("method", "endpoint"),
    [
        ("POST", "start_experiment_ert"),
        ("DELETE", "register"),
        ("POST", "runpath"),
        ("DELETE", "runpath"),
        ("POST", "runmodel"),
        ("GET", "failed_realizations"),
    ],
)
@pytest.mark.parametrize("other_experiment", [False, True])
def test_that_config_id_requests_reject_missing_or_non_ert_states(
    monkeypatch: pytest.MonkeyPatch,
    runner_client: TestClient,
    method: str,
    endpoint: str,
    other_experiment: bool,
) -> None:
    config_id = str(uuid4())
    if other_experiment:
        monkeypatch.setitem(_experiments, config_id, ExperimentRunnerState())

    response = runner_client.request(
        method, f"/experiment_runs/{endpoint}", params={"config_id": config_id}
    )

    assert response.status_code == 404
    assert response.json() == {"detail": f"ERT config '{config_id}' not found"}


def test_that_discard_closes_storage_and_removes_the_registration(
    monkeypatch: pytest.MonkeyPatch, runner_client: TestClient
) -> None:
    model = Mock(spec=RunModel)
    model._storage = Mock()
    state = ert_runner.ErtExperimentRunnerState(
        run_model=model, status_queue=SimpleQueue()
    )
    monkeypatch.setitem(_experiments, "registered-config", state)

    response = runner_client.delete(
        "/experiment_runs/register", params={"config_id": "registered-config"}
    )

    assert response.status_code == 200
    assert response.content == b""
    model._storage.close.assert_called_once_with()
    assert "registered-config" not in _experiments
    assert (
        runner_client.delete(
            "/experiment_runs/register", params={"config_id": "registered-config"}
        ).status_code
        == 404
    )
    model._storage.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_that_start_rejects_a_state_resolved_before_discard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock(spec=RunModel)
    model._storage = Mock()
    state = ert_runner.ErtExperimentRunnerState(
        run_model=model, status_queue=SimpleQueue()
    )
    monkeypatch.setitem(_experiments, "registered-config", state)
    resolved_state = ert_runner._get_ert_experiment("registered-config")
    await ert_runner.discard_registration("registered-config")
    tasks = BackgroundTasks()

    with pytest.raises(ert_runner.HTTPException) as error:
        await ert_runner.start_experiment_ert(resolved_state, tasks)

    assert error.value.status_code == 404
    assert not tasks.tasks


@pytest.mark.parametrize("experiment_status", list(ExperimentState))
def test_that_discard_rejects_started_experiments_without_closing_storage(
    monkeypatch: pytest.MonkeyPatch,
    runner_client: TestClient,
    experiment_status: ExperimentState,
) -> None:
    model = Mock(spec=RunModel)
    model._storage = Mock()
    state = ert_runner.ErtExperimentRunnerState(
        run_model=model,
        status_queue=SimpleQueue(),
        start_time_unix=1,
        status=ExperimentStatus(status=experiment_status),
    )
    monkeypatch.setitem(_experiments, "registered-config", state)

    response = runner_client.delete(
        "/experiment_runs/register", params={"config_id": "registered-config"}
    )

    assert response.status_code == 409
    model._storage.close.assert_not_called()
    assert _experiments["registered-config"] is state


def test_that_a_registered_model_cannot_be_started_twice(
    monkeypatch: pytest.MonkeyPatch,
    runner_client: TestClient,
) -> None:
    config_id = str(uuid4())
    state = ert_runner.ErtExperimentRunnerState(
        run_model=Mock(spec=RunModel), status_queue=SimpleQueue()
    )
    monkeypatch.setitem(_experiments, config_id, state)
    run_ert = AsyncMock()
    monkeypatch.setattr(ert_runner, "run_ert", run_ert)

    response = runner_client.post(
        "/experiment_runs/start_experiment_ert", params={"config_id": config_id}
    )
    assert response.status_code == 200
    assert response.content == b""

    response = runner_client.post(
        "/experiment_runs/start_experiment_ert", params={"config_id": config_id}
    )
    assert response.status_code == 409
    run_ert.assert_awaited_once_with(state, rerun_failed_realizations=False)


@pytest.mark.parametrize(
    "terminal_status", [ExperimentState.completed, ExperimentState.failed]
)
def test_that_rerun_reuses_the_model_and_clears_previous_events(
    monkeypatch: pytest.MonkeyPatch,
    runner_client: TestClient,
    terminal_status: ExperimentState,
) -> None:
    model = Mock(spec=RunModel)
    model.supports_rerunning_failed_realizations = True
    model.has_failed_realizations.return_value = True
    state = ert_runner.ErtExperimentRunnerState(
        run_model=model,
        status_queue=SimpleQueue(),
        start_time_unix=1,
        status=ExperimentStatus(status=terminal_status),
        events=[EndEvent(failed=True, msg="Previous run")],
    )
    monkeypatch.setitem(_experiments, "registered-config", state)
    run_ert = AsyncMock()
    monkeypatch.setattr(ert_runner, "run_ert", run_ert)

    response = runner_client.post(
        "/experiment_runs/start_experiment_ert",
        params={"config_id": "registered-config", "rerun_failed_realizations": True},
    )

    assert response.status_code == 200
    assert response.content == b""
    assert state.events == []
    assert state.run_model is model
    assert state.status.status == ExperimentState.pending
    run_ert.assert_awaited_once_with(state, rerun_failed_realizations=True)


@pytest.mark.parametrize(
    ("status", "supports_rerun", "has_failures"),
    [
        (ExperimentState.pending, True, True),
        (ExperimentState.running, True, True),
        (ExperimentState.completed, False, True),
        (ExperimentState.completed, True, False),
    ],
)
def test_that_rerun_rejects_unfinished_or_ineligible_models(
    monkeypatch: pytest.MonkeyPatch,
    runner_client: TestClient,
    status: ExperimentState,
    supports_rerun: bool,
    has_failures: bool,
) -> None:
    model = Mock(spec=RunModel)
    model.supports_rerunning_failed_realizations = supports_rerun
    model.has_failed_realizations.return_value = has_failures
    state = ert_runner.ErtExperimentRunnerState(
        run_model=model,
        status_queue=SimpleQueue(),
        status=ExperimentStatus(status=status),
    )
    monkeypatch.setitem(_experiments, "registered-config", state)
    run_ert = AsyncMock()
    monkeypatch.setattr(ert_runner, "run_ert", run_ert)

    response = runner_client.post(
        "/experiment_runs/start_experiment_ert",
        params={"config_id": "registered-config", "rerun_failed_realizations": True},
    )

    assert response.status_code == 409
    run_ert.assert_not_awaited()
