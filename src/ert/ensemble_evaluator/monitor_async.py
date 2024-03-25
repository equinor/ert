import asyncio
import logging
import pickle
import ssl
import uuid
from typing import TYPE_CHECKING, List, Optional

from cloudevents.conversion import to_json
from cloudevents.exceptions import DataUnmarshallerError
from cloudevents.http import CloudEvent, from_json
from websockets import Headers, WebSocketClientProtocol
from websockets.client import connect

from ert.ensemble_evaluator import identifiers
from ert.serialization import evaluator_marshaller, evaluator_unmarshaller

if TYPE_CHECKING:
    from .evaluator_connection_info import EvaluatorConnectionInfo


logger = logging.getLogger(__name__)


class MonitorAsync:
    def __init__(self, ee_con_info: "EvaluatorConnectionInfo") -> None:
        self._ee_con_info = ee_con_info
        # self._ws_duplexer: Optional[SyncWebsocketDuplexer] = None
        self._id = str(uuid.uuid1()).split("-", maxsplit=1)[0]
        self._events: asyncio.Queue[CloudEvent] = asyncio.Queue()
        self._msg_gueue: asyncio.Queue[CloudEvent] = asyncio.Queue()
        self._connection: Optional[WebSocketClientProtocol] = None
        self._monitor_tasks: List[asyncio.Task[None]] = []
        self._connected: asyncio.Event = asyncio.Event()

    async def __aenter__(self) -> "MonitorAsync":
        self._monitor_tasks = [
            asyncio.create_task(self._publisher()),
            asyncio.create_task(self._receiver()),
        ]
        await self._connected.wait()
        print("DEBUG connected!!!!!!")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        for task in self._monitor_tasks:
            task.cancel()
        results = await asyncio.gather(*self._monitor_tasks, return_exceptions=True)
        for result in results or []:
            if not isinstance(result, asyncio.CancelledError) and isinstance(
                result, Exception
            ):
                logger.error(str(result))
                raise result

    def get_base_uri(self) -> str:
        return self._ee_con_info.url

    async def _publisher(self):
        tls: Optional[ssl.SSLContext] = None
        if self._ee_con_info.cert:
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            tls.load_verify_locations(cadata=self._ee_con_info.cert)
        headers = Headers()
        if self._ee_con_info.token:
            headers["token"] = self._ee_con_info.token

        async for conn in connect(
            self._ee_con_info.client_uri,
            ssl=tls,
            extra_headers=headers,
            max_size=2**26,
            max_queue=500,
            open_timeout=5,
            ping_timeout=60,
            ping_interval=60,
            close_timeout=60,
        ):
            self._connection = conn
            self._connected.set()
            while True:
                event = await self._events.get()
                await conn.send(to_json(event, data_marshaller=evaluator_marshaller))

    async def signal_cancel(self) -> None:
        logger.debug(f"monitor-{self._id} asking server to cancel...")

        out_cloudevent = CloudEvent(
            {
                "type": identifiers.EVTYPE_EE_USER_CANCEL,
                "source": f"/ert/monitor/{self._id}",
                "id": str(uuid.uuid1()),
            }
        )
        await self._events.put(out_cloudevent)
        logger.debug(f"monitor-{self._id} asked server to cancel")

    async def signal_done(self) -> None:
        logger.debug(f"monitor-{self._id} informing server monitor is done...")

        out_cloudevent = CloudEvent(
            {
                "type": identifiers.EVTYPE_EE_USER_DONE,
                "source": f"/ert/monitor/{self._id}",
                "id": str(uuid.uuid1()),
            }
        )
        await self._events.put(out_cloudevent)
        logger.debug(f"monitor-{self._id} informed server monitor is done")

    async def _receiver(self) -> None:
        for message in self._connection:
            try:
                event = from_json(
                    str(message), data_unmarshaller=evaluator_unmarshaller
                )
            except DataUnmarshallerError:
                event = from_json(str(message), data_unmarshaller=pickle.loads)
            await self._msg_gueue.put(event)
            if event["type"] == identifiers.EVTYPE_EE_TERMINATED:
                logger.debug(f"monitor-{self._id} client received terminated")
                break

    # async def track(self) -> AsyncGenerator[CloudEvent, None]:
    async def track(self) -> CloudEvent:
        msg = await self._msg_gueue.get()
        return msg
