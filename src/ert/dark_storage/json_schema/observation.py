from uuid import UUID
from typing import List, Optional, Any, Mapping
from pydantic import BaseModel


class _ObservationTransformation(BaseModel):
    name: str
    active: List[bool]
    scale: List[float]
    observation_id: UUID


class ObservationTransformationIn(_ObservationTransformation):
    pass


class ObservationTransformationOut(_ObservationTransformation):
    id: UUID

    class Config:
        orm_mode = True


class _Observation(BaseModel):
    name: str
    errors: List[float]
    values: List[float]
    x_axis: List[Any]
    records: Optional[List[UUID]] = None


class ObservationIn(_Observation):
    pass


class ObservationOut(_Observation):
    id: UUID
    transformation: Optional[ObservationTransformationOut] = None
    userdata: Mapping[str, Any] = {}

    class Config:
        orm_mode = True
