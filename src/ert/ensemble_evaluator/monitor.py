import logging
import pickle
import uuid
from contextlib import ExitStack
from typing import TYPE_CHECKING, Optional

from cloudevents.conversion import to_json
from cloudevents.exceptions import DataUnmarshallerError
from cloudevents.http import CloudEvent, from_json

from ert.ensemble_evaluator import identifiers
from ert.serialization import evaluator_marshaller, evaluator_unmarshaller

from .sync_ws_duplexer import SyncWebsocketDuplexer

if TYPE_CHECKING:
    from .ensemble_evaluator import EvaluatorConnectionInfo


logger = logging.getLogger(__name__)


class Monitor:
    def __init__(self, ee_con_info: "EvaluatorConnectionInfo") -> None:
        self._ee_con_info = ee_con_info
        self._ws_duplexer: Optional[SyncWebsocketDuplexer] = None
        self._id = str(uuid.uuid1()).split("-", maxsplit=1)[0]

    def __enter__(self):
        self._ws_duplexer = SyncWebsocketDuplexer(
            self._ee_con_info.client_uri,
            self._ee_con_info.url,
            self._ee_con_info.cert,
            self._ee_con_info.token,
        )
        return self

    def __exit__(self, *args):
        self._ws_duplexer.stop()

    def get_base_uri(self):
        return self._ee_con_info.url

    def _send_event(self, cloud_event: CloudEvent) -> None:
        with ExitStack() as stack:
            duplexer = self._ws_duplexer
            if not duplexer:
                duplexer = SyncWebsocketDuplexer(
                    self._ee_con_info.client_uri,
                    self._ee_con_info.url,
                    self._ee_con_info.cert,
                    self._ee_con_info.token,
                )
                stack.callback(duplexer.stop)
            duplexer.send(to_json(cloud_event, data_marshaller=evaluator_marshaller))

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
        logger.debug(f"monitor-{self._id} informed server monitor is done")

    def track(self):
        with ExitStack() as stack:
            duplexer = self._ws_duplexer
            if not duplexer:
                duplexer = SyncWebsocketDuplexer(
                    self._ee_con_info.client_uri,
                    self._ee_con_info.url,
                    self._ee_con_info.cert,
                    self._ee_con_info.token,
                )
                stack.callback(duplexer.stop)
            for message in duplexer.receive():
                try:
                    event = from_json(message, data_unmarshaller=evaluator_unmarshaller)
                except DataUnmarshallerError:
                    event = from_json(message, data_unmarshaller=pickle.loads)
                yield event
                if event["type"] == identifiers.EVTYPE_EE_TERMINATED:
                    logger.debug(f"monitor-{self._id} client received terminated")
                    break
