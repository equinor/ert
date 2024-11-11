import logging
import socket
import uuid
import warnings

import zmq

from ert.shared import find_available_socket
from ert.shared import get_machine_name as ert_shared_get_machine_name

from .evaluator_connection_info import EvaluatorConnectionInfo

logger = logging.getLogger(__name__)


def get_machine_name() -> str:
    warnings.warn(
        "get_machine_name has been moved from ert.ensemble_evaluator.config to ert.shared",
        DeprecationWarning,
        stacklevel=2,
    )
    return ert_shared_get_machine_name()


class EvaluatorServerConfig:
    def __init__(
        self,
        custom_port_range: range | None = None,
        use_token: bool = True,
        custom_host: str | None = None,
        use_ipc_protocol: bool = True,
    ) -> None:
        self.host: str | None = None
        self.router_port: int | None = None
        self.url = f"ipc:///tmp/socket-{uuid.uuid4().hex[:8]}"
        self.token: str | None = None
        self._socket_handle: socket.socket | None = None

        self.server_public_key: bytes | None = None
        self.server_secret_key: bytes | None = None
        if not use_ipc_protocol:
            self._socket_handle = find_available_socket(
                custom_range=custom_port_range,
                custom_host=custom_host,
                will_close_then_reopen_socket=True,
            )
            self.host, self.router_port = self._socket_handle.getsockname()
            self.url = f"tcp://{self.host}:{self.router_port}"

        if use_token:
            self.server_public_key, self.server_secret_key = zmq.curve_keypair()
            self.token = self.server_public_key.decode("utf-8")

    def get_socket(self) -> socket.socket | None:
        if self._socket_handle:
            return self._socket_handle.dup()
        return None

    def get_connection_info(self) -> EvaluatorConnectionInfo:
        return EvaluatorConnectionInfo(
            self.url,
            self.token,
        )
