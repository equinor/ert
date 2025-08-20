import asyncio
import dataclasses
import logging
import os
import queue
import time
import traceback
import uuid
from base64 import b64decode
from functools import partial
from queue import SimpleQueue

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocketException,
)
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette import status
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.websockets import WebSocket

from ert.config import QueueSystem
from ert.ensemble_evaluator import EndEvent, EvaluatorServerConfig
from ert.run_models import StatusEvents
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig
from everest.detached.everserver import (
    ExperimentState,
    ExperimentStatus,
    _get_optimization_status,
)
from everest.strings import EVERSERVER, EverEndpoints

router = APIRouter(prefix="/experiment_server", tags=["experiment_server"])


class UserCancelled(Exception):
    pass


@dataclasses.dataclass
class ExperimentRunnerState:
    status: ExperimentStatus = dataclasses.field(default_factory=ExperimentStatus)
    events: list[StatusEvents] = dataclasses.field(default_factory=list)
    subscribers: dict[str, "Subscriber"] = dataclasses.field(default_factory=dict)
    config_path: str | os.PathLike[str] | None = None
    start_time_unix: int | None = None


shared_data = ExperimentRunnerState()
security = HTTPBasic()


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
    logging.getLogger(EVERSERVER).info(
        f"{request.scope['path']} entered from "
        f"{request.client.host if request.client else 'unknown host'} "
        f"with HTTP {request.method}"
    )


@router.get("/")
def get_status(
    request: Request, credentials: HTTPBasicCredentials = Depends(security)
) -> PlainTextResponse:
    _log(request)
    _check_user(credentials)
    return PlainTextResponse("Everest is running")


@router.get("/status")
def experiment_status(
    request: Request, credentials: HTTPBasicCredentials = Depends(security)
) -> ExperimentStatus:
    _log(request)
    _check_user(credentials)
    return shared_data.status


@router.post("/" + EverEndpoints.stop)
def stop(
    request: Request, credentials: HTTPBasicCredentials = Depends(security)
) -> Response:
    _log(request)
    _check_user(credentials)
    shared_data.status = ExperimentStatus(
        message="Server stopped by user", status=ExperimentState.stopped
    )
    return Response("Raise STOP flag succeeded. Everest initiates shutdown..", 200)


@router.post("/" + EverEndpoints.start_experiment)
async def start_experiment(
    request: Request,
    background_tasks: BackgroundTasks,
    credentials: HTTPBasicCredentials = Depends(security),
) -> Response:
    _log(request)
    _check_user(credentials)
    if shared_data.status.status == ExperimentState.pending:
        request_data = await request.json()
        config = EverestConfig.with_plugins(request_data)
        runner = ExperimentRunner(config)
        try:
            background_tasks.add_task(runner.run)
            shared_data.status = ExperimentStatus(
                status=ExperimentState.running, message="Experiment started"
            )
            # Assume only one unique running experiment per everserver instance
            # Ideally, we should return the experiment ID in the response here
            shared_data.config_path = config.config_path

            # Assume client and server is always in the same timezone
            # so disregard timestamps
            shared_data.start_time_unix = int(time.time())
            return Response("Everest experiment started")
        except Exception as e:
            shared_data.status = ExperimentStatus(
                status=ExperimentState.failed,
                message=f"Could not start experiment: {e!s}",
            )
            logging.getLogger(EVERSERVER).exception(e)
            return Response(f"Could not start experiment: {e!s}", status_code=501)
    return Response("Everest experiment is running")


@router.get("/" + EverEndpoints.config_path)
async def config_path(
    request: Request, credentials: HTTPBasicCredentials = Depends(security)
) -> Response:
    _log(request)
    _check_user(credentials)
    if shared_data.status.status == ExperimentState.pending:
        return Response("No experiment started", status_code=404)

    return Response(str(shared_data.config_path), status_code=200)


@router.get("/" + EverEndpoints.start_time)
async def start_time(
    request: Request, credentials: HTTPBasicCredentials = Depends(security)
) -> Response:
    _log(request)
    _check_user(credentials)
    if shared_data.status.status == ExperimentState.pending:
        return Response("No experiment started", status_code=404)

    return Response(str(shared_data.start_time_unix), status_code=200)


@router.websocket("/events")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _check_authentication(websocket.headers.get("Authorization"))
    subscriber_id = str(uuid.uuid4())
    try:
        while True:
            event = await _get_event(subscriber_id=subscriber_id)
            await websocket.send_json(jsonable_encoder(event))
            if isinstance(event, EndEvent):
                break
    except Exception as e:
        logging.getLogger(EVERSERVER).exception(str(e))
    finally:
        logging.getLogger(EVERSERVER).info(
            f"Subscriber {subscriber_id} done. Closing websocket"
        )
        # Give some time for subscribers to get events
        await asyncio.sleep(5)
        shared_data.subscribers[subscriber_id].done()


async def _get_event(subscriber_id: str) -> StatusEvents:
    """
    The function waits until there is an event available for the subscriber
    and returns the event. If the subscriber is up to date it will
    wait until we wake up the subscriber using notify
    """
    if subscriber_id not in shared_data.subscribers:
        shared_data.subscribers[subscriber_id] = Subscriber()
    subscriber = shared_data.subscribers[subscriber_id]

    while subscriber.index >= len(shared_data.events):
        await subscriber.wait_for_event()

    event = shared_data.events[subscriber.index]
    subscriber.index += 1
    return event


class ExperimentRunner:
    def __init__(
        self,
        everest_config: EverestConfig,
    ) -> None:
        super().__init__()

        self._everest_config = everest_config

    async def run(self) -> None:
        status_queue: SimpleQueue[StatusEvents] = SimpleQueue()
        try:
            run_model = EverestRunModel.create(
                self._everest_config,
                optimization_callback=partial(_opt_monitor, shared_data=shared_data),
                status_queue=status_queue,
            )
            shared_data.status = ExperimentStatus(
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
                if shared_data.status.status == ExperimentState.stopped:
                    run_model.cancel()
                    raise UserCancelled("Optimization aborted")
                try:
                    item: StatusEvents = status_queue.get(block=False)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                shared_data.events.append(item)
                for sub in shared_data.subscribers.values():
                    sub.notify()

                if isinstance(item, EndEvent):
                    # Wait for subscribers to receive final events
                    for sub in shared_data.subscribers.values():
                        await sub.is_done()
                    break
            await simulation_future
            assert run_model.exit_code is not None
            exp_status, msg = _get_optimization_status(
                run_model.exit_code,
                shared_data.events,
            )
            shared_data.status = ExperimentStatus(
                message=msg,
                status=exp_status,
            )
        except UserCancelled as e:
            logging.getLogger(EVERSERVER).exception(e)
        except Exception as e:
            logging.getLogger(EVERSERVER).exception(e)
            shared_data.status = ExperimentStatus(
                message=f"Exception: {e}\n{traceback.format_exc()}",
                status=ExperimentState.failed,
            )
        finally:
            logging.getLogger(EVERSERVER).info(
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


def _opt_monitor(shared_data: ExperimentRunnerState) -> str | None:
    if shared_data.status.status.stopped:
        return "stop_optimization"
    return None
