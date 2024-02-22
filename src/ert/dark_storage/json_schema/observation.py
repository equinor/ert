from typing import Any, List, Mapping, Optional
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass
class _ObservationTransformation:
    name: str
    active: List[bool]
    scale: List[float]
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
    errors: List[float]
    values: List[float]
    x_axis: List[Any]
    records: Optional[List[UUID]] = None


@dataclass
class ObservationIn(_Observation):
    pass


@dataclass(config=ConfigDict(from_attributes=True))
class ObservationOut(_Observation):
    id: UUID = Field(default_factory=uuid4)
    transformation: Optional[ObservationTransformationOut] = None
    userdata: Mapping[str, Any] = Field(default_factory=dict)
