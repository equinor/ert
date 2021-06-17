import logging
import pickle
import uuid
from contextlib import ExitStack
from typing import Optional, TYPE_CHECKING

import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from cloudevents.exceptions import DataUnmarshallerError
from cloudevents.http import from_json, to_json
from cloudevents.http.event import CloudEvent
from ert_shared.ensemble_evaluator.entity import serialization
from ert_shared.ensemble_evaluator.sync_ws_duplexer import SyncWebsocketDuplexer

if TYPE_CHECKING:
    from ert.ensemble_evaluator import EvaluatorConnectionInfo


logger = logging.getLogger(__name__)


class _BaseMonitor:
    def __init__(
        self,
        ee_con_info: "EvaluatorConnectionInfo",
        endpoint,
        end_event_type,
    ):
        self._ee_con_info = ee_con_info
        self._end_event_type = end_event_type
        self._endpoint = endpoint
        self._ws_duplexer: Optional[SyncWebsocketDuplexer] = None
        self._id = str(uuid.uuid1()).split("-")[0]

    def __enter__(self):
        self._ws_duplexer = SyncWebsocketDuplexer(
            f"{self._ee_con_info.url}/{self._endpoint}",
            self._ee_con_info.url,
            self._ee_con_info.cert,
            self._ee_con_info.token,
        )
        return self

    def __exit__(self, *args):
        self._ws_duplexer.stop()

    def get_base_uri(self):
        return self._ee_con_info.url

    def get_client_uri(self):
        return self._client_uri

    def _send_event(self, cloud_event: CloudEvent) -> None:
        with ExitStack() as stack:
            duplexer = self._ws_duplexer
            if not duplexer:
                duplexer = SyncWebsocketDuplexer(
                    f"{self._ee_con_info.url}/{self._endpoint}",
                    self._ee_con_info.url,
                    self._ee_con_info.cert,
                    self._ee_con_info.token,
                )
                stack.callback(duplexer.stop)
            duplexer.send(
                to_json(cloud_event, data_marshaller=serialization.evaluator_marshaller)
            )

    def track(self):
        with ExitStack() as stack:
            duplexer = self._ws_duplexer
            if not duplexer:
                duplexer = SyncWebsocketDuplexer(
                    f"{self._ee_con_info.url}/{self._endpoint}",
                    self._ee_con_info.url,
                    self._ee_con_info.cert,
                    self._ee_con_info.token,
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
                if event["type"] == self._end_event_type:
                    logger.debug(f"monitor-{self._id} client received terminated")
                    break


class _Monitor(_BaseMonitor):
    def __init__(self, evaluation_id, ee_con_info: "EvaluatorConnectionInfo"):
        self._evaluation_id = evaluation_id
        super().__init__(
            ee_con_info=ee_con_info,
            endpoint=f"client/{self._evaluation_id}",
            end_event_type=identifiers.EVTYPE_EE_TERMINATED,
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


class _ExperimentMonitor(_BaseMonitor):
    def __init__(self, experiment_id, ee_con_info: "EvaluatorConnectionInfo"):
        self._experiment_id = experiment_id
        super().__init__(
            ee_con_info,
            endpoint=f"experiment/{self._experiment_id}",
            end_event_type=identifiers.EVTYPE_EE_EXPERIMENT_TERMINATED,
        )


def create(evaluation_id: str, ee_con_info: "EvaluatorConnectionInfo") -> _Monitor:
    return _Monitor(evaluation_id, ee_con_info)


def create_experiment(
    experiment_id: str, ee_con_info: "EvaluatorConnectionInfo"
) -> _ExperimentMonitor:
    return _ExperimentMonitor(experiment_id, ee_con_info)
