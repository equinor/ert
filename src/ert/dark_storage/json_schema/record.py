from uuid import UUID
from typing import Any, Mapping
from pydantic import BaseModel, Field


class _Record(BaseModel):
    pass


class RecordOut(_Record):
    id: UUID
    name: str
    userdata: Mapping[str, Any]

    class Config:
        orm_mode = True
