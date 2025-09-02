import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ert.config.queue_config import (
    LocalQueueOptions,
    LsfQueueOptions,
    SlurmQueueOptions,
    TorqueQueueOptions,
)
from ert.dark_storage.client import ConnInfo

from ..strings import (
    CERTIFICATE_DIR,
    DETACHED_NODE_DIR,
    HOSTFILE_NAME,
    SERVER_STATUS,
    SESSION_DIR,
)
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
        description="Defines which queue system the everest server is submitted to",
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
    def get_server_url(output_dir: str) -> str:
        """Return the url of the server.

        If server_info are given, the url is generated using that info. Otherwise
        server information are retrieved from the hostfile
        """
        server_info = ServerConfig.get_server_info(output_dir)
        return f"https://{server_info['host']}:{server_info['port']}/experiment_server"

    @staticmethod
    def get_server_context_from_conn_info(
        conn_info: ConnInfo,
    ) -> tuple[str, str, tuple[str, str]]:
        """Get server connection context information from a storage_service ConnInfo.

        Returns a tuple containing the server URL, certificate file path,
        and authentication credentials.
        NOTE: This function is to temporarily bridge the gap between ERT storage client
        and the Everest clients setup + requesting.
        Currently Everest client side uses the get_server_context and requests directly.
        This should be refactored to use the ERT Storage Client class directly instead.

        Args:
            conn_info: An instance of the storage_service client ConnInfo

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
    def get_server_info(output_dir: str) -> dict[str, Any]:
        """Load server information from the hostfile"""
        host_file_path = Path(ServerConfig.get_hostfile_path(output_dir))
        if not host_file_path.exists():
            return {"host": None, "port": None, "cert": None, "auth": None}

        data = json.loads(host_file_path.read_text(encoding="utf-8"))

        if not all(k in data for k in ("host", "port", "cert", "auth")):
            raise RuntimeError("Malformed hostfile")
        return data

    @staticmethod
    def get_detached_node_dir(output_dir: str) -> str:
        return os.path.join(os.path.abspath(output_dir), DETACHED_NODE_DIR)

    @staticmethod
    def get_hostfile_path(output_dir: str) -> str:
        return os.path.join(ServerConfig.get_session_dir(output_dir), HOSTFILE_NAME)

    @staticmethod
    def get_session_dir(output_dir: str) -> str:
        """Return path to the session directory containing information about the
        certificates and host information"""
        return os.path.join(ServerConfig.get_detached_node_dir(output_dir), SESSION_DIR)

    @staticmethod
    def get_everserver_status_path(output_dir: str) -> str:
        """Returns path to the everest server status file"""
        return os.path.join(ServerConfig.get_session_dir(output_dir), SERVER_STATUS)

    @staticmethod
    def get_certificate_dir(output_dir: str) -> str:
        """Return the path to certificate folder"""
        return os.path.join(ServerConfig.get_session_dir(output_dir), CERTIFICATE_DIR)
