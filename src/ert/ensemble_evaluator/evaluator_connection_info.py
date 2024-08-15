from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class EvaluatorConnectionInfo:
    """Read only server-info"""

    url: str
    cert: Optional[Union[str, bytes]] = None
    token: Optional[str] = None

    @property
    def dispatch_uri(self) -> str:
        return f"{self.url}/dispatch"

    @property
    def client_uri(self) -> str:
        return f"{self.url}/client"

    @property
    def result_uri(self) -> str:
        return f"{self.url}/result"
