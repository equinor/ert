import asyncio
import dataclasses
import datetime
import json
import logging
import os
import queue
import shutil
import signal
import time
import traceback
import uuid
import warnings
from base64 import b64decode
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from queue import SimpleQueue
from typing import Annotated, Any, cast

import anyio
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocketException,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, ConfigDict
from starlette import status
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.websockets import WebSocket

from _ert.threading import ErtThread
from ert.base_model_context import use_runtime_plugins
from ert.config import ConfigWarning, QueueSystem
from ert.config.ert_config import ErtConfig
from ert.dark_storage.common import EverEndpoints
from ert.ensemble_evaluator import EndEvent, EvaluatorServerConfig
from ert.ensemble_evaluator.event import FullSnapshotEvent, SnapshotUpdateEvent
from ert.ensemble_evaluator.snapshot import EnsembleSnapshot
from ert.namespace import Namespace
from ert.plugins import get_site_plugins
from ert.run_models import StatusEvents
from ert.run_models.everest_run_model import EverestExitCode, EverestRunModel
from ert.run_models.model_factory import create_model
from ert.run_models.run_model import RunModel
from everest.config import EverestConfig
from everest.everserver.server import (
    ExperimentState,
    ExperimentStatus,
)
from everest.strings import (
    OPT_FAILURE_ALL_REALIZATIONS,
    OPT_FAILURE_REALIZATIONS,
)

router = APIRouter(prefix="/experiment_runs", tags=["experiment_runs"])


class UserCancelled(Exception):
    pass


@dataclasses.dataclass
class ExperimentRunnerState:
    status: ExperimentStatus = dataclasses.field(default_factory=ExperimentStatus)
    events: list[StatusEvents] = dataclasses.field(default_factory=list)
    subscribers: dict[str, "Subscriber"] = dataclasses.field(default_factory=dict)
    config_path: str | os.PathLike[str] | None = None
    run_path: str | os.PathLike[str] | None = None
    storage_path: str | os.PathLike[str] | None = None
    start_time_unix: int | None = None


_experiments: dict[str, ExperimentRunnerState] = {}


class RunArgs(BaseModel):
    model_config = ConfigDict(extra="allow")
    mode: str


class StartErtExperimentRequest(BaseModel):
    config: ErtConfig
    args: RunArgs


def _get_experiment(experiment_id: str) -> ExperimentRunnerState:
    if experiment_id not in _experiments:
        raise HTTPException(
            status_code=404, detail=f"Experiment '{experiment_id}' not found"
        )
    return _experiments[experiment_id]


def _failed_realizations_messages(
    events: list[StatusEvents], exit_code: EverestExitCode
) -> list[str]:
    snapshots: dict[int, EnsembleSnapshot] = {}
    for event in events:
        if isinstance(event, FullSnapshotEvent) and event.snapshot:
            snapshots[event.iteration] = event.snapshot
        elif isinstance(event, SnapshotUpdateEvent) and event.snapshot:
            snapshot = snapshots[event.iteration]
            snapshot.merge_snapshot(event.snapshot)
    messages = [
        OPT_FAILURE_REALIZATIONS
        if exit_code == EverestExitCode.TOO_FEW_REALIZATIONS
        else OPT_FAILURE_ALL_REALIZATIONS
    ]
    for snapshot in snapshots.values():
        for job in snapshot.get_all_fm_steps().values():
            if error := job.get("error"):
                msg = f"{job.get('name', 'Unknown name')} Failed with: {error}"
                if msg not in messages:
                    messages.append(msg)
    return messages


def _get_optimization_status(
    exit_code: EverestExitCode | None, events: list[StatusEvents]
) -> tuple[ExperimentState, str]:
    match exit_code:
        case EverestExitCode.MAX_BATCH_NUM_REACHED:
            return ExperimentState.completed, "Maximum number of batches reached."

        case EverestExitCode.MAX_FUNCTIONS_REACHED:
            return (
                ExperimentState.completed,
                "Maximum number of function evaluations reached.",
            )

        case EverestExitCode.USER_ABORT:
            return ExperimentState.stopped, "Optimization aborted."

        case (
            EverestExitCode.TOO_FEW_REALIZATIONS
            | EverestExitCode.ALL_REALIZATIONS_FAILED
        ):
            status_ = ExperimentState.failed
            messages = _failed_realizations_messages(events, exit_code)
            for msg in messages:
                logging.getLogger(__name__).error(msg)
            return status_, "\n".join(messages)
        case EverestExitCode.COMPLETED:
            return ExperimentState.completed, "Optimization completed."
        case _:
            raise ValueError(f"Invalid exit_code: {exit_code}")


def _check_authentication(auth_header: str | None) -> None:
    if auth_header is None:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION, reason="No authentication"
        )
    _, encoded_credentials = auth_header.split(" ")
    decoded_credentials = b64decode(encoded_credentials).decode("utf-8")
    _, _, password = decoded_credentials.partition(":")
    if password != os.environ["ERT_STORAGE_TOKEN"]:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)


def verify_auth(
    request: Request,
    credentials: Annotated[HTTPBasicCredentials, Depends(HTTPBasic())],
) -> None:
    logging.getLogger(__name__).debug(
        f"{request.scope['path']} entered from "
        f"{request.client.host if request.client else 'unknown host'} "
        f"with HTTP {request.method}"
    )
    if credentials.password != os.environ["ERT_STORAGE_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


authenticated = [Depends(verify_auth)]


async def _with_runtime_plugins() -> AsyncIterator[None]:
    stack = ExitStack()
    try:
        stack.enter_context(warnings.catch_warnings())
        warnings.filterwarnings("ignore", category=ConfigWarning)
        stack.enter_context(use_runtime_plugins(get_site_plugins()))
        yield
    finally:
        stack.close()


@router.get("/", dependencies=authenticated)
def get_status() -> PlainTextResponse:
    return PlainTextResponse("EVEREST is running")


@router.get(f"/{EverEndpoints.STATUS}/{{experiment_id}}", dependencies=authenticated)
def experiment_status(
    experiment: Annotated[ExperimentRunnerState, Depends(_get_experiment)],
) -> ExperimentStatus:
    return experiment.status


@router.get("/" + EverEndpoints.EXPERIMENTS, dependencies=authenticated)
def experiments() -> JSONResponse:
    return JSONResponse({"experiment_ids": list(_experiments.keys())})


@router.post("/" + EverEndpoints.STOP, dependencies=authenticated)
def stop() -> Response:
    if not _experiments:
        os.kill(os.getpid(), signal.SIGTERM)
    for experiment in _experiments.values():
        experiment.status = ExperimentStatus(
            message="Server stopped by user", status=ExperimentState.stopped
        )
    return Response("Raise STOP flag succeeded. EVEREST initiates shutdown..", 200)


@router.post(
    "/" + EverEndpoints.START_EXPERIMENT,
    dependencies=[*authenticated, Depends(_with_runtime_plugins)],
)
async def start_experiment(
    config: EverestConfig,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    experiment_id = str(uuid.uuid4())
    experiment_state = ExperimentRunnerState()
    _experiments[experiment_id] = experiment_state
    try:
        background_tasks.add_task(run_everest, config, experiment_id)
        experiment_state.config_path = config.config_path
        experiment_state.run_path = config.simulation_dir
        experiment_state.storage_path = config.output_dir
        experiment_state.start_time_unix = int(time.time())
        return JSONResponse({"experiment_id": experiment_id})
    except Exception:
        error_message = "Could not start experiment due to an internal error."
        experiment_state.status = ExperimentStatus(
            status=ExperimentState.failed,
            message=error_message,
        )
        logging.getLogger(__name__).exception("Failed to start experiment")
        return JSONResponse({"error": error_message}, status_code=501)


@router.post(
    "/" + EverEndpoints.START_EXPERIMENT_ERT,
    dependencies=[*authenticated, Depends(_with_runtime_plugins)],
)
async def start_experiment_ert(
    request: StartErtExperimentRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    experiment_id = str(uuid.uuid4())
    experiment_state = ExperimentRunnerState()
    _experiments[experiment_id] = experiment_state
    ert_config = request.config
    try:
        background_tasks.add_task(run_ert, ert_config, request.args, experiment_id)
        experiment_state.config_path = ert_config.config_path
        experiment_state.run_path = ert_config.runpath_file
        experiment_state.storage_path = ert_config.ens_path
        experiment_state.start_time_unix = int(time.time())
        return JSONResponse({"experiment_id": experiment_id})
    except Exception:
        error_message = "Could not start experiment due to an internal error."
        experiment_state.status = ExperimentStatus(
            status=ExperimentState.failed,
            message=error_message,
        )
        logging.getLogger(__name__).exception("Failed to start experiment")
        return JSONResponse({"error": error_message}, status_code=501)


@router.get(
    f"/{EverEndpoints.CONFIG_PATH}/{{experiment_id}}", dependencies=authenticated
)
async def config_path(
    experiment: Annotated[ExperimentRunnerState, Depends(_get_experiment)],
) -> JSONResponse:
    if experiment.status.status == ExperimentState.pending:
        return JSONResponse("No experiment started", status_code=404)

    return JSONResponse(
        {
            "config_path": str(experiment.config_path),
            "run_path": str(experiment.run_path),
            "storage_path": str(experiment.storage_path),
        },
        status_code=200,
    )


@router.get(
    f"/{EverEndpoints.START_TIME}/{{experiment_id}}", dependencies=authenticated
)
async def start_time(
    experiment: Annotated[ExperimentRunnerState, Depends(_get_experiment)],
) -> Response:
    if experiment.status.status == ExperimentState.pending:
        return Response("No experiment started", status_code=404)

    return Response(str(experiment.start_time_unix), status_code=200)


@router.post(
    f"/{EverEndpoints.RUNPATH}",
    dependencies=[*authenticated, Depends(_with_runtime_plugins)],
)
async def check_runpath_exists(request: StartErtExperimentRequest) -> Response:
    """
    Check if any of the given paths (iteration directories) exists.
    Returns a 200 response if at least one path exists, 404 otherwise.
    """
    exists = False
    model = create_model(request.config, cast(Namespace, request.args), SimpleQueue())
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        async with anyio.create_task_group() as tg:

            async def _check_path(path: str) -> None:
                nonlocal exists
                if await anyio.Path(path).exists():
                    exists = True
                    tg.cancel_scope.cancel()

            for path in model.paths:
                tg.start_soon(_check_path, path)
    except Exception as e:
        logging.getLogger(__name__).exception(str(e))
    finally:
        model._storage.close()

    if exists:
        return Response("Runpath exists", status_code=200)
    return Response("Runpath does not exist", status_code=404)


@router.delete(
    f"/{EverEndpoints.RUNPATH}",
    dependencies=[*authenticated, Depends(_with_runtime_plugins)],
)
def delete_runpath(request: StartErtExperimentRequest) -> Response:
    model = create_model(request.config, cast(Namespace, request.args), SimpleQueue())

    def delete_path(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)

    try:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(delete_path, Path(path)) for path in model.paths]
            for future in futures:
                future.result()
    except Exception as e:
        logging.getLogger(__name__).exception(str(e))
        return Response("Failed to delete runpaths", status_code=500)
    finally:
        model._storage.close()
    return Response("All runpaths deleted", status_code=200)


@router.post("/runmodel", dependencies=[*authenticated, Depends(_with_runtime_plugins)])
def get_runmodel_data(request: StartErtExperimentRequest) -> Response:
    model = create_model(request.config, cast(Namespace, request.args), SimpleQueue())
    data = {
        "number_of_existing_runpaths": model.get_number_of_existing_runpaths(),
        "number_of_active_realizations": model.get_number_of_active_realizations(),
    }
    model._storage.close()
    return Response(json.dumps(data), status_code=200)


@router.websocket(f"/{EverEndpoints.EVENTS}/{{experiment_id}}")
async def websocket_endpoint(websocket: WebSocket, experiment_id: str) -> None:
    await websocket.accept()
    _check_authentication(websocket.headers.get("Authorization"))
    if experiment_id not in _experiments:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    subscriber_id = str(uuid.uuid4())
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        while True:
            event = await _get_event(
                subscriber_id=subscriber_id, experiment_id=experiment_id
            )
            await websocket.send_json(jsonable_encoder(event))
            if isinstance(event, EndEvent):
                await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                break
    except Exception as e:
        logging.getLogger(__name__).exception(str(e))
    finally:
        logging.getLogger(__name__).info(
            f"Subscriber {subscriber_id} done. Closing websocket"
        )
        # Give some time for subscribers to get events
        await asyncio.sleep(5)
        _experiments[experiment_id].subscribers[subscriber_id].done()


async def _get_event(subscriber_id: str, experiment_id: str) -> StatusEvents:
    """
    The function waits until there is an event available for the subscriber
    and returns the event. If the subscriber is up to date it will
    wait until we wake up the subscriber using notify
    """
    run = _experiments[experiment_id]
    if subscriber_id not in run.subscribers:
        run.subscribers[subscriber_id] = Subscriber()
    subscriber = run.subscribers[subscriber_id]

    while subscriber.index >= len(run.events):
        await subscriber.wait_for_event()

    event = run.events[subscriber.index]
    subscriber.index += 1
    return event


async def run_everest(config: EverestConfig, experiment_id: str) -> None:
    run = _experiments[experiment_id]
    status_queue: SimpleQueue[StatusEvents] = SimpleQueue()
    run_model: EverestRunModel | None = None
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        site_plugins = get_site_plugins()
        with use_runtime_plugins(site_plugins):
            run_model = EverestRunModel.create(
                everest_config=config,
                experiment_name=f"EnOpt@{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}",
                target_ensemble="batch",
                status_queue=status_queue,
                runtime_plugins=site_plugins,
            )
        run.status = ExperimentStatus(
            message="Experiment started", status=ExperimentState.running
        )
        loop = asyncio.get_running_loop()
        simulation_future = loop.run_in_executor(
            None,
            lambda: run_model.start_simulations_thread(
                EvaluatorServerConfig()
                if run_model.queue_config.queue_system == QueueSystem.LOCAL
                else EvaluatorServerConfig(use_ipc_protocol=False)
            ),
        )
        while True:
            if run.status.status == ExperimentState.stopped:
                run_model.cancel()
                raise UserCancelled("Optimization aborted")
            try:
                item: StatusEvents = status_queue.get(block=False)
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            run.events.append(item)
            for sub in run.subscribers.values():
                sub.notify()

            if isinstance(item, EndEvent):
                # Wait for subscribers to receive final events
                for sub in list(run.subscribers.values()):
                    await sub.is_done()
                break
        await simulation_future
        exp_status, msg = _get_optimization_status(
            run_model.exit_code,
            run.events,
        )
        run.status = ExperimentStatus(
            message=msg,
            status=exp_status,
        )
    except UserCancelled as e:
        logging.getLogger(__name__).info(f"User cancelled: {e}")
    except Exception as e:
        logging.getLogger(__name__).exception(e)
        run.status = ExperimentStatus(
            message=f"Exception: {e}\n{traceback.format_exc()}",
            status=ExperimentState.failed,
        )
    finally:
        if run_model and run_model._experiment:
            run_model._experiment.status = run.status

        logging.getLogger(__name__).info(
            f"ExperimentRunner done. Items left in queue: {status_queue.qsize()}"
        )


async def run_ert(config: ErtConfig, args: Any, experiment_id: str) -> None:
    run = _experiments[experiment_id]
    status_queue: SimpleQueue[StatusEvents] = SimpleQueue()
    cancellation_requested = False

    def publish(event: StatusEvents) -> None:
        run.events.append(event)
        for subscriber in run.subscribers.values():
            subscriber.notify()

    try:  # ruff: ignore[too-many-statements-in-try-clause]
        if run.status.status == ExperimentState.stopped:
            publish(EndEvent(failed=True, msg="Experiment cancelled before start."))
            return

        model = create_model(config, args, status_queue)
        run.status = ExperimentStatus(
            message="Experiment started", status=ExperimentState.running
        )
        evaluator_config = EvaluatorServerConfig(
            use_ipc_protocol=model.queue_config.queue_system == QueueSystem.LOCAL
        )
        simulation_future = asyncio.get_running_loop().run_in_executor(
            None, model.start_simulations_thread, evaluator_config
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


def start_ert_simulation_thread(
    model: RunModel,
    queue_system: QueueSystem,
    *,
    rerun_failed_realizations: bool = False,
) -> None:
    simulation_thread = get_simulation_thread(
        model,
        rerun_failed_realizations=rerun_failed_realizations,
        use_ipc_protocol=queue_system == QueueSystem.LOCAL,
    )
    simulation_thread.start()


def get_simulation_thread(
    model: RunModel,
    *,
    rerun_failed_realizations: bool = False,
    use_ipc_protocol: bool = False,
) -> ErtThread:
    evaluator_server_config = EvaluatorServerConfig(use_ipc_protocol=use_ipc_protocol)

    def run() -> None:
        model.api.start_simulations_thread(
            evaluator_server_config=evaluator_server_config,
            rerun_failed_realizations=rerun_failed_realizations,
        )

    return ErtThread(name="ert_server_simulation_thread", target=run, daemon=True)


class Subscriber:
    """
    This class keeps track of events and allows subscribers
    to wait for new events to occur. Each subscriber instance
    can be notified of an event, at which point any coroutines
    that are waiting for an event will resume execution.
    """

    def __init__(self) -> None:
        self.index = 0
        self._event = asyncio.Event()
        self._done = asyncio.Event()

    def notify(self) -> None:
        self._event.set()

    def done(self) -> None:
        self._done.set()

    async def wait_for_event(self) -> None:
        await self._event.wait()
        self._event.clear()

    async def is_done(self) -> None:
        await self._done.wait()
