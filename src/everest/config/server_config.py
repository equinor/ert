from pathlib import Path
from textwrap import dedent
from typing import Any

from pydantic import BaseModel, Field, NonNegativeInt, model_validator

from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
)
from ert.services.shared_client import ErtClientConnectionInfo
from everest.strings import SESSION_DIR

from .simulator_config import check_removed_config


class ServerConfig(BaseModel, extra="forbid"):
    queue_system: (
        LocalQueueOptions
        | LsfQueueOptions
        | SlurmQueueOptions
        | TorqueQueueOptions
        | None
    ) = Field(
        default=None,
        description=dedent(
            """
            Defines which queue system the everest server is submitted to.

            For example, to run the server locally even if the forward model may
            run on another system:

            ```yaml
            server:
              queue_system:
                name: local
            ```
            """
        ),
        discriminator="name",
    )
    max_runtime: NonNegativeInt | None = Field(
        default=None,
        description=dedent(
            """
            Maximum allowed running time for the everest server job, specified
            in seconds.

            The limit, padded with ten minutes, is passed on to the queue system
            the server is submitted to (e.g. `bsub -W` for LSF). When the limit
            is reached, the queue system stops the server and with it the whole
            optimization. If not set, or set to 0, the default limit of the queue
            system applies. For Slurm, this overrides `max_runtime` given in
            `server.queue_system`. The `max_runtime` of the simulator's Slurm
            queue system is not applied to the server. This option has no
            effect on the local queue.
            """
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def check_old_config(cls, data: Any) -> Any:
        if isinstance(data, dict):
            check_removed_config(data.get("queue_system"))
        return data

    @staticmethod
    def get_server_context_from_conn_info(
        conn_info: ErtClientConnectionInfo,
    ) -> tuple[str, str, tuple[str, str]]:
        """Get server connection context information from a ErtClientConnectionInfo.

        Returns a tuple containing the server URL, certificate file path,
        and authentication credentials.
        NOTE: This function is to temporarily bridge the gap between ERT storage client
        and the EVEREST clients setup + requesting.
        Currently EVEREST client side uses the get_server_context and requests directly.
        This should be refactored to use the ERT Storage Client class directly instead.

        Args:
            conn_info: An instance of the ErtClientConnectionInfo

        Returns:
            tuple: A tuple containing:
                - str: URL of the server
                - str: Path to the certificate file
                - tuple[str, str]: Username and password for authentication
        """
        url = conn_info.base_url + "/experiment_runs"
        cert_file = conn_info.cert
        auth_token = conn_info.auth_token
        if auth_token is None:
            raise RuntimeError("No authentication token found in storage session")
        auth = ("username", auth_token)
        if not isinstance(cert_file, str):
            raise RuntimeError("Invalid certificate file in storage session")

        return url, cert_file, auth

    @staticmethod
    def get_session_dir(output_dir: str) -> str:
        """Return path to the session directory containing information about the
        certificates and host information
        """
        return str(Path(output_dir).resolve() / SESSION_DIR)
