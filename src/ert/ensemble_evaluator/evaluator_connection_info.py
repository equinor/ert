from dataclasses import dataclass


@dataclass
class EvaluatorConnectionInfo:
    """Read only server-info"""

    router_uri: str
    token: str | None = None
