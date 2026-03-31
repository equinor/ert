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
    EXPERIMENT_SERVER,
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


_runs: dict[str, ExperimentRunnerState] = {}
security = HTTPBasic()


def _get_run(run_id: str) -> ExperimentRunnerState:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return _runs[run_id]


def _failed_realizations_messages(
    events: list[StatusEvents], exit_code: EverestExitCode
) -> list[str]:
    snapshots: dict[int, EnsembleSnapshot] = {}
    for event in events:
        if isinstance(event, FullSnapshotEvent) and event.snapshot:
            snapshots[event.iteration] = event.snapshot
        elif isinstance(event, SnapshotUpdateEvent) and event.snapshot:
            snapshot = snapshots[event.iteration]
            assert isinstance(snapshot, EnsembleSnapshot)
            snapshot.merge_snapshot(event.snapshot)
    logging.getLogger("forward_models").info("Status event")
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
    exit_code: EverestExitCode, events: list[StatusEvents]
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
                logging.getLogger(EXPERIMENT_SERVER).error(msg)
            return status_, "\n".join(messages)
        case _:
            return ExperimentState.completed, "Optimization completed."


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


def _check_user(credentials: HTTPBasicCredentials) -> None:
    if credentials.password != os.environ["ERT_STORAGE_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


def _log(request: Request) -> None:
    logging.getLogger(EXPERIMENT_SERVER).debug(
        f"{request.scope['path']} entered from "
        f"{request.client.host if request.client else 'unknown host'} "
        f"with HTTP {request.method}"
    )


@router.get("/")
def get_status(
    request: Request, credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> PlainTextResponse:
    _log(request)
    _check_user(credentials)
    return PlainTextResponse("EVEREST is running")


@router.get(f"/{EverEndpoints.status}/{{run_id}}")
def experiment_status(
    request: Request,
    run: Annotated[ExperimentRunnerState, Depends(_get_run)],
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> ExperimentStatus:
    _log(request)
    _check_user(credentials)
    return run.status


@router.get("/runs")
def runs(
    request: Request,
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> JSONResponse:
    _log(request)
    _check_user(credentials)
    return JSONResponse({"run_ids": list(_runs.keys())})


@router.post("/" + EverEndpoints.stop)
def stop(
    request: Request,
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> Response:
    _log(request)
    _check_user(credentials)
    if not _runs:
        os.kill(os.getpid(), signal.SIGTERM)
    for run in _runs.values():
        run.status = ExperimentStatus(
            message="Server stopped by user", status=ExperimentState.stopped
        )
    return Response("Raise STOP flag succeeded. EVEREST initiates shutdown..", 200)


@router.post("/" + EverEndpoints.start_experiment)
async def start_experiment(
    request: Request,
    background_tasks: BackgroundTasks,
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> JSONResponse:
    _log(request)
    _check_user(credentials)
    run_id = str(uuid.uuid4())
    run_state = ExperimentRunnerState()
    _runs[run_id] = run_state
    request_data = await request.json()
    # The output of warnings is the task of the user interface, not
    # of everserver. Therefore we suppress them here:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConfigWarning)
        config = EverestConfig.with_plugins(request_data)
    runner = ExperimentRunner(config, run_id)
    try:
        background_tasks.add_task(runner.run)
        run_state.config_path = config.config_path

        run_state.run_path = config.simulation_dir
        run_state.storage_path = config.output_dir

        # Assume client and server is always in the same timezone
        # so disregard timestamps
        run_state.start_time_unix = int(time.time())
        return JSONResponse({"run_id": run_id})
    except Exception as e:
        run_state.status = ExperimentStatus(
            status=ExperimentState.failed,
            message=f"Could not start experiment: {e!s}",
        )
        logging.getLogger(EXPERIMENT_SERVER).exception(e)
        return JSONResponse(
            {"error": f"Could not start experiment: {e!s}"}, status_code=501
        )


@router.get(f"/{EverEndpoints.config_path}/{{run_id}}")
async def config_path(
    request: Request,
    run: Annotated[ExperimentRunnerState, Depends(_get_run)],
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> JSONResponse:
    _log(request)
    _check_user(credentials)
    if run.status.status == ExperimentState.pending:
        return JSONResponse("No experiment started", status_code=404)

    return JSONResponse(
        {
            "config_path": str(run.config_path),
            "run_path": str(run.run_path),
            "storage_path": str(run.storage_path),
        },
        status_code=200,
    )


@router.get(f"/{EverEndpoints.start_time}/{{run_id}}")
async def start_time(
    request: Request,
    run: Annotated[ExperimentRunnerState, Depends(_get_run)],
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> Response:
    _log(request)
    _check_user(credentials)
    if run.status.status == ExperimentState.pending:
        return Response("No experiment started", status_code=404)

    return Response(str(run.start_time_unix), status_code=200)


@router.websocket(f"/{EverEndpoints.events}/{{run_id}}")
async def websocket_endpoint(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    _check_authentication(websocket.headers.get("Authorization"))
    if run_id not in _runs:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    subscriber_id = str(uuid.uuid4())
    try:
        while True:
            event = await _get_event(subscriber_id=subscriber_id, run_id=run_id)
            await websocket.send_json(jsonable_encoder(event))
            if isinstance(event, EndEvent):
                break
    except Exception as e:
        logging.getLogger(EXPERIMENT_SERVER).exception(str(e))
    finally:
        logging.getLogger(EXPERIMENT_SERVER).info(
            f"Subscriber {subscriber_id} done. Closing websocket"
        )
        # Give some time for subscribers to get events
        await asyncio.sleep(5)
        _runs[run_id].subscribers[subscriber_id].done()


async def _get_event(subscriber_id: str, run_id: str) -> StatusEvents:
    """
    The function waits until there is an event available for the subscriber
    and returns the event. If the subscriber is up to date it will
    wait until we wake up the subscriber using notify
    """
    run = _runs[run_id]
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
        run_id: str,
    ) -> None:
        super().__init__()

        self._everest_config = everest_config
        self._run_id = run_id

    async def run(self) -> None:
        run = _runs[self._run_id]
        status_queue: SimpleQueue[StatusEvents] = SimpleQueue()
        run_model: EverestRunModel | None = None
        try:
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
            assert run_model.exit_code is not None
            exp_status, msg = _get_optimization_status(
                run_model.exit_code,
                run.events,
            )
            run.status = ExperimentStatus(
                message=msg,
                status=exp_status,
            )
        except UserCancelled as e:
            logging.getLogger(EXPERIMENT_SERVER).info(f"User cancelled: {e}")
        except Exception as e:
            logging.getLogger(EXPERIMENT_SERVER).exception(e)
            run.status = ExperimentStatus(
                message=f"Exception: {e}\n{traceback.format_exc()}",
                status=ExperimentState.failed,
            )
        finally:
            if run_model and run_model._experiment:
                run_model._experiment.status = run.status

            logging.getLogger(EXPERIMENT_SERVER).info(
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
