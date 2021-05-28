from uuid import UUID

from pydantic import BaseModel
from typing import List, Union, Optional
from .observation import (
    ObservationTransformationIn,
)


class _Update(BaseModel):
    algorithm: str
    ensemble_result_id: Union[UUID, None]
    ensemble_reference_id: Union[UUID, None]


class UpdateIn(_Update):
    observation_transformations: Optional[List[ObservationTransformationIn]] = None


class UpdateOut(_Update):
    id: UUID
    experiment_id: UUID

    class Config:
        orm_mode = True
