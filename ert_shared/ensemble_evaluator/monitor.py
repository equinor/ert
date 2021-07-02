import logging
import pickle
import uuid
from contextlib import ExitStack
from typing import Optional

import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from cloudevents.exceptions import DataUnmarshallerError
from cloudevents.http import from_json, to_json
from cloudevents.http.event import CloudEvent
from ert_shared.ensemble_evaluator.entity import serialization
from ert_shared.ensemble_evaluator.sync_ws_duplexer import SyncWebsocketDuplexer


logger = logging.getLogger(__name__)


class _Monitor:
    def __init__(self, host, port, protocol="wss", cert=None, token=None):
        self._base_uri = f"{protocol}://{host}:{port}"
        self._client_uri = f"{self._base_uri}/client"
        self._result_uri = f"{self._base_uri}/result"
        self._cert = cert
        self._token = token
        self._ws_duplexer: Optional[SyncWebsocketDuplexer] = None
        self._id = str(uuid.uuid1()).split("-")[0]

    def __enter__(self):
        self._ws_duplexer = SyncWebsocketDuplexer(
            self._client_uri, self._base_uri, self._cert, self._token
        )
        return self

    def __exit__(self, *args):
        self._ws_duplexer.stop()

    def get_base_uri(self):
        return self._base_uri

    def _send_event(self, cloud_event: CloudEvent) -> None:
        with ExitStack() as stack:
            duplexer = self._ws_duplexer
            if not duplexer:
                duplexer = SyncWebsocketDuplexer(
                    self._client_uri, self._base_uri, self._cert, self._token
                )
                stack.callback(duplexer.stop)
            duplexer.send(
                to_json(cloud_event, data_marshaller=serialization.evaluator_marshaller)
            )

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
                    self._client_uri, self._base_uri, self._cert, self._token
                )
                stack.callback(duplexer.stop)
            for message in duplexer.receive():
                try:
                    event = from_json(
                        message, data_unmarshaller=serialization.evaluator_unmarshaller
                    )
                except DataUnmarshallerError:
                    event = from_json(message, data_unmarshaller=pickle.loads)
                yield event
                if event["type"] == identifiers.EVTYPE_EE_TERMINATED:
                    logger.debug(f"monitor-{self._id} client received terminated")
                    break


def create(host, port, protocol, cert, token):
    return _Monitor(host, port, protocol, cert, token)
