from typing import Any, Mapping, Optional
from uuid import UUID

from pydantic import BaseModel


class _Record(BaseModel):
    pass


class RecordOut(_Record):
    id: UUID
    name: str
    userdata: Mapping[str, Any]
    has_observations: Optional[bool]

    class Config:
        from_attributes = True
