from collections.abc import Mapping
from typing import Any
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass
class _ObservationTransformation:
    name: str
    active: list[bool]
    scale: list[float]
    observation_id: UUID


@dataclass
class ObservationTransformationIn(_ObservationTransformation):
    pass


@dataclass(config=ConfigDict(from_attributes=True))
class ObservationTransformationOut(_ObservationTransformation):
    id: UUID


@dataclass
class _Observation:
    name: str
    errors: list[float]
    values: list[float]
    x_axis: list[Any]
    records: list[UUID] | None = None


@dataclass
class ObservationIn(_Observation):
    pass


@dataclass(config=ConfigDict(from_attributes=True))
class ObservationOut(_Observation):
    id: UUID = Field(default_factory=uuid4)
    transformation: ObservationTransformationOut | None = None
    userdata: Mapping[str, Any] = Field(default_factory=dict)
