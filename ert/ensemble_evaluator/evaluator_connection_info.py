from typing import Optional, Union


class EvaluatorConnectionInfo:
    """Read only server-info"""

    def __init__(
        self,
        host: str,
        port: int,
        url: str,
        cert: Optional[Union[str, bytes]] = None,
        token: Optional[str] = None,
    ) -> None:
        # pylint: disable=too-many-arguments
        self.host = host
        self.port = port
        self.cert = cert
        self.token = token
        self.url = url

    @property
    def dispatch_uri(self) -> str:
        return f"{self.url}/dispatch"

    @property
    def client_uri(self) -> str:
        return f"{self.url}/client"

    @property
    def result_uri(self) -> str:
        return f"{self.url}/result"
