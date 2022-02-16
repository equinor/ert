from uuid import UUID
from typing import Any, Mapping, List
from pydantic import BaseModel, Field


class _Record(BaseModel):
    pass


class RecordOut(_Record):
    id: UUID
    name: str
    userdata: Mapping[str, Any]
    observations: List[Any]

    class Config:
        orm_mode = True
