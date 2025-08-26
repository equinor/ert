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
        description="Defines which queue system the everest submits jobs to",
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
    def get_server_context(output_dir: str) -> tuple[str, str, tuple[str, str]]:
        """Returns a tuple with
        - url of the server
        - path to the .cert file
        - password for the certificate file
        """
        server_info = ServerConfig.get_server_info(output_dir)
        return (
            ServerConfig.get_server_url(output_dir),
            server_info[CERTIFICATE_DIR],
            ("username", server_info["auth"]),
        )

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
