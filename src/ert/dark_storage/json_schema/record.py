from typing import Any, Mapping, Optional
from uuid import UUID

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass
class _Record:
    pass


@dataclass(config=ConfigDict(from_attributes=True))
class RecordOut(_Record):
    id: UUID
    name: str
    userdata: Mapping[str, Any]
    has_observations: Optional[bool]


@dataclass(config=ConfigDict(from_attributes=True))
class ImageOut(_Record):
    image: bytes
