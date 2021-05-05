import asyncio
import websockets
from websockets.datastructures import Headers
from ert_shared.ensemble_evaluator.utils import wait_for_evaluator
import logging
import threading
from cloudevents.http import from_json
from cloudevents.http.event import CloudEvent
from cloudevents.http import to_json
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from ert_shared.ensemble_evaluator.entity import serialization
import uuid
import pickle
import ssl
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ert3.data import RecordTransmitter


logger = logging.getLogger(__name__)


class _Monitor:
    def __init__(self, host, port, protocol="wss", cert=None, token=None):
        self._base_uri = f"{protocol}://{host}:{port}"
        self._client_uri = f"{self._base_uri}/client"
        self._result_uri = f"{self._base_uri}/result"
        self._token = token
        self._extra_headers = Headers()
        if token is not None:
            self._extra_headers["token"] = token

        # Mimics the behavior of the ssl argument when connection to
        # websockets. If none is specified it will deduce based on the url,
        # if True it will enforce TLS, and if you want to use self signed
        # certificates you need to pass an ssl_context with the certificate
        # loaded.
        self._cert = cert
        if cert is not None:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_context.load_verify_locations(cadata=cert)
        else:
            ssl_context = True if protocol == "wss" else None
        self._ssl_context = ssl_context

        self._loop = None
        self._incoming = None
        self._receive_future = None
        self._id = str(uuid.uuid1()).split("-")[0]

    def __enter__(self):
        self._loop = asyncio.new_event_loop()
        self._incoming = asyncio.Queue(loop=self._loop)
        return self

    def __exit__(self, *args):
        self._loop.close()

    def get_base_uri(self):
        return self._base_uri

    def get_result(self) -> Dict[int, Dict[str, "RecordTransmitter"]]:
        async def _send():
            async with websockets.connect(
                self._result_uri,
                ssl=self._ssl_context,
                extra_headers=self._extra_headers,
            ) as websocket:
                result = await websocket.recv()
                message = from_json(result, lambda x: pickle.loads(x))
                return message.data

        return asyncio.run_coroutine_threadsafe(_send(), self._loop).result()

    def _send_event(self, cloud_event):
        async def _send():
            async with websockets.connect(
                self._client_uri,
                ssl=self._ssl_context,
                extra_headers=self._extra_headers,
            ) as websocket:
                message = to_json(
                    cloud_event, data_marshaller=serialization.evaluator_marshaller
                )
                await websocket.send(message)

        asyncio.run_coroutine_threadsafe(_send(), self._loop).result()

    def signal_cancel(self):
        logger.debug(f"monitor-{self._id} asking server to cancel...")

        out_cloudevent = CloudEvent(
            {
                "type": identifiers.EVTYPE_EE_USER_CANCEL,
                "source": f"/ert/monitor/{self._id}",
                "id": str(uuid.uuid1()),
            }
        )
        self._send_event(out_cloudevent)
        logger.debug(f"monitor-{self._id} asked server to cancel")

    def signal_done(self):
        logger.debug(f"monitor-{self._id} informing server monitor is done...")

        out_cloudevent = CloudEvent(
            {
                "type": identifiers.EVTYPE_EE_USER_DONE,
                "source": f"/ert/monitor/{self._id}",
                "id": str(uuid.uuid1()),
            }
        )
        self._send_event(out_cloudevent)
        logger.debug(f"monitor-{self._id} informing server monitor is done...")

    async def _receive(self):
        logger.debug(f"monitor-{self._id} starting receive")
        async with websockets.connect(
            self._client_uri,
            ssl=self._ssl_context,
            extra_headers=self._extra_headers,
            max_size=2 ** 26,
            max_queue=500,
        ) as websocket:
            async for message in websocket:
                event = from_json(
                    message, data_unmarshaller=serialization.evaluator_unmarshaller
                )
                self._incoming.put_nowait(event)
                if event["type"] == identifiers.EVTYPE_EE_TERMINATED:
                    logger.debug(f"monitor-{self._id} client received terminated")
                    break

        logger.debug(f"monitor-{self._id} disconnected")

    def _run(self, done_future):
        asyncio.set_event_loop(self._loop)
        self._receive_future = self._loop.create_task(self._receive())
        try:
            self._loop.run_until_complete(self._receive_future)
        except asyncio.CancelledError:
            logger.debug(f"monitor-{self._id} receive cancelled")
        self._loop.run_until_complete(done_future)

    def track(self):
        asyncio.get_event_loop().run_until_complete(
            wait_for_evaluator(
                base_url=self._base_uri, token=self._token, cert=self._cert
            )
        )

        done_future = asyncio.Future(loop=self._loop)

        thread = threading.Thread(
            name=f"ert_monitor-{self._id}_loop", target=self._run, args=(done_future,)
        )
        thread.start()

        event = None
        try:
            while event is None or event["type"] != identifiers.EVTYPE_EE_TERMINATED:
                event = asyncio.run_coroutine_threadsafe(
                    self._incoming.get(), self._loop
                ).result()
                yield event
            self._loop.call_soon_threadsafe(done_future.set_result, None)
        except GeneratorExit:
            logger.debug(f"monitor-{self._id} generator exit")
            self._loop.call_soon_threadsafe(self._receive_future.cancel)
            if not done_future.done():
                self._loop.call_soon_threadsafe(done_future.set_result, None)
        thread.join()


def create(host, port, protocol, cert, token):
    return _Monitor(host, port, protocol, cert, token)
