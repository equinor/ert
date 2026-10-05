import asyncio
import dataclasses
import json
import logging
import queue
import time
import uuid
from functools import partial
from queue import SimpleQueue
from typing import Annotated, Literal, Self, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator
from starlette.responses import Response

from ert.config import QueueSystem
from ert.config.ert_config import ErtConfig
from ert.config.everest_control import EverestControl
from ert.config.field import Field
from ert.config.gen_kw_config import GenKwConfig
from ert.config.surface_config import SurfaceConfig
from ert.ensemble_evaluator import EndEvent, EvaluatorServerConfig
from ert.namespace import Namespace
from ert.run_models import StatusEvents
from ert.run_models.model_factory import create_model
from ert.run_models.run_model import RunModel
from ert.server.common import ErtRunnerEndpoints, get_storage
from everest.everserver.server import ExperimentState, ExperimentStatus

from .experiment_runs import (
    ExperimentRunnerState,
    _experiments,
    _with_runtime_plugins,
    authenticated,
)

router = APIRouter(prefix="/experiment_runs", tags=["experiment_runs"])


@dataclasses.dataclass(kw_only=True)
class ErtExperimentRunnerState(ExperimentRunnerState):
    run_model: RunModel
    status_queue: SimpleQueue[StatusEvents]


class RunArgs(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Literal[
        "test_run",
        "ensemble_experiment",
        "ensemble_smoother",
        "ensemble_information_filter",
        "es_mda",
        "evaluate_ensemble",
        "manual_update",
        "manual_enif_update",
    ]
    experiment_name: str | None = None
    current_ensemble: str = "default"
    target_ensemble: str | None = None
    ensemble_id: str | None = None
    prior_ensemble_id: str | None = None
    realizations: str | None = None
    ensemble_size: int | None = None
    weights: str | None = None

    @model_validator(mode="after")
    def validate_mode_arguments(self) -> Self:
        required: list[str] = []

        if self.mode not in {"test_run", "evaluate_ensemble"}:
            required.append("experiment_name")

        if self.mode in {
            "ensemble_smoother",
            "ensemble_information_filter",
            "manual_update",
            "manual_enif_update",
        }:
            required.append("target_ensemble")

        if self.mode in {"evaluate_ensemble", "manual_update", "manual_enif_update"}:
            required.append("ensemble_id")

        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(f"{self.mode} requires: {', '.join(missing)}")
        return self

    @model_validator(mode="after")
    def deserialize_parameter_configuration(self) -> Self:
        if (
            self.model_extra is not None
            and "parameter_configuration" in self.model_extra
        ):
            self.model_extra["parameter_configuration"] = TypeAdapter(
                list[GenKwConfig | Field | SurfaceConfig | EverestControl]
            ).validate_python(self.model_extra["parameter_configuration"])
        return self


class StartErtExperimentRequest(BaseModel):
    config: ErtConfig
    args: RunArgs


class ConfigIdRequest(BaseModel):
    config_id: str


def _get_ert_experiment(config_id: str) -> ErtExperimentRunnerState:
    state = _experiments.get(config_id)
    if not isinstance(state, ErtExperimentRunnerState):
        raise HTTPException(
            status_code=404, detail=f"ERT config '{config_id}' not found"
        )
    return state


@router.post(
    "/" + ErtRunnerEndpoints.REGISTER,
    dependencies=[*authenticated, Depends(_with_runtime_plugins)],
)
async def register(request: StartErtExperimentRequest) -> JSONResponse:
    config_id = str(uuid.uuid4())
    status_queue: SimpleQueue[StatusEvents] = SimpleQueue()
    run_model = create_model(
        request.config, cast(Namespace, request.args), status_queue
    )
    experiment_state = ErtExperimentRunnerState(
        run_model=run_model,
        status_queue=status_queue,
        config_path=request.config.config_path,
        run_path=request.config.runpath_file,
        storage_path=request.config.ens_path,
    )
    _experiments[config_id] = experiment_state
    return JSONResponse({"config_id": config_id})


@router.delete(
    "/" + ErtRunnerEndpoints.REGISTER,
    dependencies=[*authenticated],
)
async def discard_registration(config_id: str) -> Response:
    config = _get_ert_experiment(config_id)
    if config.start_time_unix is not None:
        raise HTTPException(status_code=409, detail="ERT experiment already started")
    config.run_model._storage.close()
    del _experiments[config_id]
    return Response(status_code=200)


@router.post(
    "/" + ErtRunnerEndpoints.START_EXPERIMENT_ERT,
    dependencies=[*authenticated],
)
async def start_experiment_ert(
    config: Annotated[ErtExperimentRunnerState, Depends(_get_ert_experiment)],
    background_tasks: BackgroundTasks,
    *,
    rerun_failed_realizations: bool = False,
) -> Response:
    if not any(registered is config for registered in _experiments.values()):
        raise HTTPException(status_code=404, detail="ERT registration was discarded")
    if rerun_failed_realizations:
        if (
            config.status.status
            not in {ExperimentState.completed, ExperimentState.failed}
            or not config.run_model.supports_rerunning_failed_realizations
            or not config.run_model.has_failed_realizations()
        ):
            raise HTTPException(
                status_code=409,
                detail="ERT experiment cannot rerun failed realizations",
            )
    elif config.start_time_unix is not None:
        raise HTTPException(status_code=409, detail="ERT experiment already started")
    try:
        background_tasks.add_task(
            run_ert, config, rerun_failed_realizations=rerun_failed_realizations
        )
        if rerun_failed_realizations:
            config.reset_for_rerun()
        config.start_time_unix = int(time.time())
        return Response(status_code=200)
    except Exception:
        error_message = "Could not start experiment due to an internal error."
        config.status = ExperimentStatus(
            status=ExperimentState.failed,
            message=error_message,
        )
        logging.getLogger(__name__).exception("Failed to start experiment")
        return JSONResponse({"error": error_message}, status_code=501)


@router.post(
    f"/{ErtRunnerEndpoints.RUNPATH}",
    dependencies=[*authenticated],
)
async def check_runpath_exists(
    config: Annotated[ErtExperimentRunnerState, Depends(_get_ert_experiment)],
) -> Response:
    """
    Check if runpath exists for a given experiment.
    Returns a 200 response if at least one path exists, 404 otherwise.
    """
    try:
        if config.run_model.check_if_runpath_exists():
            return Response("Runpath exists", status_code=200)
    except Exception as e:
        logging.getLogger(__name__).exception(str(e))
        raise HTTPException(
            status_code=500, detail="Error occurred while checking runpath existence"
        ) from e
    return Response("Runpath does not exist", status_code=404)


@router.delete(
    f"/{ErtRunnerEndpoints.RUNPATH}",
    dependencies=[*authenticated],
)
def delete_runpath(
    config: Annotated[ErtExperimentRunnerState, Depends(_get_ert_experiment)],
) -> Response:
    if config.start_time_unix is not None:
        raise HTTPException(
            status_code=409, detail="Cannot delete runpath while experiment is running"
        )
    try:
        config.run_model.rm_runpath()
    except Exception as e:
        logging.getLogger(__name__).exception(str(e))
        return Response("Failed to delete runpaths", status_code=500)
    return Response("All runpaths deleted", status_code=200)


@router.post(f"/{ErtRunnerEndpoints.RUNMODEL}", dependencies=[*authenticated])
def get_runmodel_data(
    config: Annotated[ErtExperimentRunnerState, Depends(_get_ert_experiment)],
) -> Response:
    model = config.run_model
    data = {
        "number_of_existing_runpaths": model.get_number_of_existing_runpaths(),
        "number_of_active_realizations": model.get_number_of_active_realizations(),
        "supports_rerunning_failed_realizations": (
            model.supports_rerunning_failed_realizations
        ),
    }
    return Response(json.dumps(data), status_code=200)


@router.get(
    f"/{ErtRunnerEndpoints.FAILED_REALIZATIONS}",
    dependencies=[*authenticated, Depends(get_storage)],
)
def get_failed_realizations(
    config: Annotated[ErtExperimentRunnerState, Depends(_get_ert_experiment)],
) -> Response:
    model = config.run_model
    data = {
        "failed_realizations": model._create_mask_from_failed_realizations(),
    }
    return Response(json.dumps(data), status_code=200)


async def run_ert(
    config: Annotated[ErtExperimentRunnerState, Depends(_get_ert_experiment)],
    *,
    rerun_failed_realizations: bool = False,
) -> None:
    run = config
    status_queue = run.status_queue
    model = run.run_model
    cancellation_requested = False

    def publish(event: StatusEvents) -> None:
        run.events.append(event)
        if isinstance(event, EndEvent):
            run.finalized.set()
        for subscriber in run.subscribers.values():
            subscriber.notify()

    try:  # ruff: ignore[too-many-statements-in-try-clause]
        if run.status.status == ExperimentState.stopped:
            model._storage.close()
            publish(EndEvent(failed=True, msg="Experiment cancelled before start."))
            return

        run.status = ExperimentStatus(
            message="Experiment started", status=ExperimentState.running
        )
        evaluator_config = EvaluatorServerConfig(
            use_ipc_protocol=model.queue_config.queue_system == QueueSystem.LOCAL
        )
        simulation_future = asyncio.get_running_loop().run_in_executor(
            None,
            partial(
                model.start_simulations_thread,
                evaluator_config,
                rerun_failed_realizations=rerun_failed_realizations,
            ),
        )

        while True:
            if (
                run.status.status == ExperimentState.stopped
                and not cancellation_requested
            ):
                model.cancel()
                cancellation_requested = True

            try:
                event = status_queue.get_nowait()
            except queue.Empty:
                if not simulation_future.done():
                    await asyncio.sleep(0.01)
                    continue
                await simulation_future
                try:
                    event = status_queue.get_nowait()
                except queue.Empty:
                    raise RuntimeError("Simulation ended without EndEvent") from None

            if isinstance(event, EndEvent):
                await simulation_future
                terminal_event = event
                break
            publish(event)

    except Exception as error:
        logging.getLogger(__name__).exception("Experiment failed")
        terminal_event = EndEvent(failed=True, msg=str(error))

    if run.status.status != ExperimentState.stopped:
        run.status = ExperimentStatus(
            message=terminal_event.msg,
            status=(
                ExperimentState.failed
                if terminal_event.failed
                else ExperimentState.completed
            ),
        )
    publish(terminal_event)
