import asyncio
import dataclasses
import datetime
import logging
import os
import queue
import signal
import time
import traceback
import uuid
import warnings
from base64 import b64decode
from queue import SimpleQueue
from typing import Annotated

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
from pydantic import BaseModel
from starlette import status
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.websockets import WebSocket

from ert.base_model_context import use_runtime_plugins
from ert.config import ConfigWarning, QueueSystem
from ert.ensemble_evaluator import EndEvent, EvaluatorServerConfig
from ert.ensemble_evaluator.event import FullSnapshotEvent, SnapshotUpdateEvent
from ert.ensemble_evaluator.snapshot import EnsembleSnapshot
from ert.plugins import get_site_plugins
from ert.run_models import StatusEvents
from ert.run_models.everest_run_model import EverestExitCode, EverestRunModel
from everest.config import EverestConfig
from everest.detached.everserver import (
    ExperimentState,
    ExperimentStatus,
)
from everest.strings import (
    OPT_FAILURE_ALL_REALIZATIONS,
    OPT_FAILURE_REALIZATIONS,
    EverEndpoints,
)

router = APIRouter(prefix="/experiment_server", tags=["experiment_server"])


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


class PathsCheckRequest(BaseModel):
    paths: list[str]


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


@router.post("/" + EverEndpoints.START_EXPERIMENT, dependencies=authenticated)
async def start_experiment(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    experiment_id = str(uuid.uuid4())
    experiment_state = ExperimentRunnerState()
    _experiments[experiment_id] = experiment_state
    request_data = await request.json()
    # Suppress already reported warnings when we re-validate with plugins
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConfigWarning)
        config = EverestConfig.with_plugins(request_data)
    runner = ExperimentRunner(config, experiment_id)
    try:
        background_tasks.add_task(runner.run)
        experiment_state.config_path = config.config_path
        experiment_state.run_path = config.simulation_dir
        experiment_state.storage_path = config.output_dir
        experiment_state.start_time_unix = int(time.time())
        return JSONResponse({"experiment_id": experiment_id})
    except Exception as e:
        experiment_state.status = ExperimentStatus(
            status=ExperimentState.failed,
            message=f"Could not start experiment: {e!s}",
        )
        logging.getLogger(__name__).exception(e)
        return JSONResponse(
            {"error": f"Could not start experiment: {e!s}"}, status_code=501
        )


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


@router.post(f"/{EverEndpoints.RUNPATH}", dependencies=authenticated)
async def check_runpath_exists(
    paths: PathsCheckRequest,
) -> Response:
    """
    Check if any of the given paths (iteration directories) exists.
    Returns a 200 response if at least one path exists, 404 otherwise.
    """
    exists = False

    async with anyio.create_task_group() as tg:

        async def _check_path(path: str) -> None:
            nonlocal exists
            if await anyio.Path(path).exists():
                exists = True
                tg.cancel_scope.cancel()

        for path in paths.paths:
            tg.start_soon(_check_path, path)

    if exists:
        return Response("Runpath exists", status_code=200)
    return Response("Runpath does not exist", status_code=404)


@router.websocket(f"/{EverEndpoints.EVENTS}/{{experiment_id}}")
async def websocket_endpoint(websocket: WebSocket, experiment_id: str) -> None:
    await websocket.accept()
    _check_authentication(websocket.headers.get("Authorization"))
    if experiment_id not in _experiments:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    subscriber_id = str(uuid.uuid4())
    try:
        while True:
            event = await _get_event(
                subscriber_id=subscriber_id, experiment_id=experiment_id
            )
            await websocket.send_json(jsonable_encoder(event))
            if isinstance(event, EndEvent):
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


class ExperimentRunner:
    def __init__(
        self,
        everest_config: EverestConfig,
        experiment_id: str,
    ) -> None:
        super().__init__()

        self._everest_config = everest_config
        self._experiment_id = experiment_id

    async def run(self) -> None:
        run = _experiments[self._experiment_id]
        status_queue: SimpleQueue[StatusEvents] = SimpleQueue()
        run_model: EverestRunModel | None = None
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            site_plugins = get_site_plugins()
            with use_runtime_plugins(site_plugins):
                run_model = EverestRunModel.create(
                    everest_config=self._everest_config,
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
