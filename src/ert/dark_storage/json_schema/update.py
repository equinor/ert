from typing import List, Optional, Union
from uuid import UUID

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from .observation import (
    ObservationTransformationIn,
)


@dataclass
class _Update:
    algorithm: str
    ensemble_result_id: Union[UUID, None]
    ensemble_reference_id: Union[UUID, None]


@dataclass
class UpdateIn(_Update):
    observation_transformations: Optional[List[ObservationTransformationIn]] = None


@dataclass(config=ConfigDict(from_attributes=True))
class UpdateOut(_Update):
    id: UUID
    experiment_id: UUID
