from uuid import UUID
from typing import Any, Mapping, Optional
from pydantic import BaseModel, Field


class _Record(BaseModel):
    pass


class RecordOut(_Record):
    id: UUID
    name: str
    userdata: Mapping[str, Any]
    has_observations: Optional[bool]

    class Config:
        orm_mode = True
