from uuid import UUID
from typing import Any, Mapping
from pydantic import BaseModel, Field


class _Record(BaseModel):
    pass


class RecordOut(_Record):
    id: UUID
    name: str
    metadata: Mapping[str, Any] = Field(alias="metadata_dict")

    class Config:
        orm_mode = True
