from typing import Optional

from pydantic import BaseModel


class ConnInfo(BaseModel):
    base_url: str
    auth_token: Optional[str] = None
