from dataclasses import dataclass


@dataclass
class EvaluatorConnectionInfo:
    """Read only server-info"""

    url: str
    cert: str | bytes | None = None
    token: str | None = None

    @property
    def dispatch_uri(self) -> str:
        return f"{self.url}/dispatch"

    @property
    def client_uri(self) -> str:
        return f"{self.url}/client"

    @property
    def result_uri(self) -> str:
        return f"{self.url}/result"
