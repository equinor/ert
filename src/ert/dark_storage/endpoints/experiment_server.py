import asyncio
import logging
import os
import time
import uuid
from base64 import b64decode
from http.client import HTTPException
from queue import SimpleQueue

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    WebSocketException,
)
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette import status
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.websockets import WebSocket

from ert.ensemble_evaluator import EndEvent
from ert.run_models import StatusEvents
from everest.config import EverestConfig
from everest.detached.jobs.everserver import (
    ExperimentRunner,
    ExperimentRunnerState,
    Subscriber,
)
from everest.strings import EVERSERVER, EverEndpoints

router = APIRouter(prefix="/experiment_server", tags=["experiment_server"])


shared_data = ExperimentRunnerState()
security = HTTPBasic()


def _check_authentication(auth_header: str) -> None:
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
) -> PlainTextResponse:
    _log(request)
    _check_user(credentials)
    if shared_data.done:
        return PlainTextResponse("Experiment is done")
    if shared_data.started:
        return PlainTextResponse("Experiment is started")


@router.post("/" + EverEndpoints.stop)
def stop(
    request: Request, credentials: HTTPBasicCredentials = Depends(security)
) -> Response:
    _log(request)
    _check_user(credentials)
    shared_data.stop = True
    # msg_queue.put(ServerStopped())
    return Response("Raise STOP flag succeeded. Everest initiates shutdown..", 200)


@router.post("/" + EverEndpoints.start_experiment)
async def start_experiment(
    request: Request,
    background_tasks: BackgroundTasks,
    credentials: HTTPBasicCredentials = Depends(security),
) -> Response:
    _log(request)
    _check_user(credentials)
    msg_queue = SimpleQueue()
    if not shared_data.started:
        request_data = await request.json()
        config = EverestConfig.with_plugins(request_data)
        runner = ExperimentRunner(config, shared_data, msg_queue)
        try:
            background_tasks.add_task(runner.run)
            shared_data.started = True

            # Assume only one unique running experiment per everserver instance
            # Ideally, we should return the experiment ID in the response here
            shared_data.config_path = config.config_path

            # Assume client and server is always in the same timezone
            # so disregard timestamps
            shared_data.start_time_unix = int(time.time())
            return Response("Everest experiment started")
        except Exception as e:
            logging.getLogger(EVERSERVER).exception(e)
            return Response(f"Could not start experiment: {e!s}", status_code=501)
    return Response("Everest experiment is running")


@router.get("/" + EverEndpoints.config_path)
async def config_path(
    request: Request, credentials: HTTPBasicCredentials = Depends(security)
) -> Response:
    _log(request)
    _check_user(credentials)
    if not shared_data.started:
        return Response("No experiment started", status_code=404)

    return Response(str(shared_data.config_path), status_code=200)


@router.get("/" + EverEndpoints.simulation_dir)
async def simulation_dir(
    request: Request, credentials: HTTPBasicCredentials = Depends(security)
) -> Response:
    _log(request)
    _check_user(credentials)
    if not shared_data.started:
        return Response("No experiment started", status_code=404)

    sim_dir = EverestConfig.from_file(shared_data.config_path).simulation_dir
    return Response(sim_dir, status_code=200)


@router.get("/" + EverEndpoints.start_time)
async def start_time(
    request: Request, credentials: HTTPBasicCredentials = Depends(security)
) -> Response:
    _log(request)
    _check_user(credentials)
    if not shared_data.started:
        return Response("No experiment started", status_code=404)

    return Response(str(shared_data.start_time_unix), status_code=200)


@router.websocket("/events")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _check_authentication(websocket.headers.get("Authorization"))
    subscriber_id = str(uuid.uuid4())
    try:
        while True:
            event = await get_event(subscriber_id=subscriber_id)
            await websocket.send_json(jsonable_encoder(event))
            if isinstance(event, EndEvent):
                shared_data.done = True
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


async def get_event(subscriber_id: str) -> StatusEvents:
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
