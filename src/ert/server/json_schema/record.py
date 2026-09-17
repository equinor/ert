from collections.abc import Mapping
from typing import Any
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
    has_observations: bool | None
