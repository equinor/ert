import logging
import uuid
import warnings

import zmq

from ert.shared import get_machine_name as ert_shared_get_machine_name
from ert.shared.net_utils import get_ip_address

logger = logging.getLogger(__name__)


def get_machine_name() -> str:
    warnings.warn(
        "get_machine_name has been moved from "
        "ert.ensemble_evaluator.config to ert.shared",
        DeprecationWarning,
        stacklevel=2,
    )
    return ert_shared_get_machine_name()


class EvaluatorServerConfig:
    def __init__(
        self,
        port_range: tuple[int, int] | None = None,
        use_token: bool = True,
        host: str | None = None,
        use_ipc_protocol: bool = True,
    ) -> None:
        self.host: str | None = host
        self.router_port: int | None = None
        self.token: str | None = None
        self.server_public_key: bytes | None = None
        self.server_secret_key: bytes | None = None
        self.use_ipc_protocol: bool = use_ipc_protocol

        if port_range is None:
            port_range = (51820, 51840 + 1)
        else:
            if port_range[0] > port_range[1]:
                raise ValueError("Minimum port in range is higher than maximum port")

            if port_range[0] == port_range[1]:
                port_range = (port_range[0], port_range[0] + 1)

        self.min_port = port_range[0]
        self.max_port = port_range[1]

        if use_ipc_protocol:
            self.uri = f"ipc:///tmp/socket-{uuid.uuid4().hex[:8]}"
        elif self.host is None:
            self.host = get_ip_address()

        if use_token:
            self.server_public_key, self.server_secret_key = zmq.curve_keypair()
            self.token = self.server_public_key.decode("utf-8")

    def get_uri(self) -> str:
        if not self.use_ipc_protocol:
            return f"tcp://{self.host}:{self.router_port}"

        return self.uri
