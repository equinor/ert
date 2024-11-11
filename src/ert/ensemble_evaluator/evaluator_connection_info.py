from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class EvaluatorConnectionInfo:
    """Read only server-info"""

    router_uri: str
    cert: Optional[Union[str, bytes]] = None
    token: Optional[str] = None
