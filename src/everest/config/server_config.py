import json
import os
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ..strings import (
    CERTIFICATE_DIR,
    DETACHED_NODE_DIR,
    HOSTFILE_NAME,
    SERVER_STATUS,
    SESSION_DIR,
)
from .has_ert_queue_options import HasErtQueueOptions


class ServerConfig(BaseModel, HasErtQueueOptions):  # type: ignore
    name: Optional[str] = Field(
        None,
        description="""Specifies which queue to use.

Examples are
* mr
* bigmem

The everest server generally has lower resource requirements than forward models such
as RMS and Eclipse.
    """,
    )  # Corresponds to queue name
    exclude_host: Optional[str] = Field(
        "",
        description="""Comma separated list of nodes that should be
         excluded from the slurm run""",
    )
    include_host: Optional[str] = Field(
        "",
        description="""Comma separated list of nodes that
        should be included in the slurm run""",
    )
    options: Optional[str] = Field(
        None,
        description="""Used to specify options to LSF.
        Examples to set memory requirement is:
        * rusage[mem=1000]""",
    )
    queue_system: Optional[Literal["lsf", "local", "slurm"]] = Field(
        None,
        description="Defines which queue system the everest server runs on.",
    )
    model_config = ConfigDict(
        extra="forbid",
    )

    @staticmethod
    def get_server_url(output_dir: str) -> str:
        """Return the url of the server.

        If server_info are given, the url is generated using that info. Otherwise
        server information are retrieved from the hostfile
        """
        server_info = ServerConfig.get_server_info(output_dir)
        return f"https://{server_info['host']}:{server_info['port']}"

    @staticmethod
    def get_server_context(output_dir: str) -> Tuple[str, str, Tuple[str, str]]:
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
    def get_server_info(output_dir: str) -> dict:
        """Load server information from the hostfile"""
        host_file_path = ServerConfig.get_hostfile_path(output_dir)
        try:
            with open(host_file_path, "r", encoding="utf-8") as f:
                json_string = f.read()

            data = json.loads(json_string)
            if set(data.keys()) != {"host", "port", "cert", "auth"}:
                raise RuntimeError("Malformed hostfile")
            return data
        except FileNotFoundError:
            # No host file
            return {"host": None, "port": None, "cert": None, "auth": None}

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
