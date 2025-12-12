import os
from textwrap import dedent
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
)
from ert.dark_storage.client import ErtClientConnectionInfo

from ..strings import SESSION_DIR
from .simulator_config import check_removed_config


class ServerConfig(BaseModel):
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
    model_config = ConfigDict(
        extra="forbid",
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
        and the Everest clients setup + requesting.
        Currently Everest client side uses the get_server_context and requests directly.
        This should be refactored to use the ERT Storage Client class directly instead.

        Args:
            conn_info: An instance of the ErtClientConnectionInfo

        Returns:
            tuple: A tuple containing:
                - str: URL of the server
                - str: Path to the certificate file
                - tuple[str, str]: Username and password for authentication
        """
        url = conn_info.base_url + "/experiment_server"
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
        certificates and host information"""
        return os.path.join(os.path.abspath(output_dir), SESSION_DIR)
