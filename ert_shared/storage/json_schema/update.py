from pydantic import BaseModel
from typing import List, Union
from ert_shared.storage.json_schema.observation_transformation import (
    ObservationTransformationCreate,
)


class UpdateBase(BaseModel):
    algorithm: str
    ensemble_result_id: Union[int, None]
    ensemble_reference_id: Union[int, None]


class UpdateCreate(UpdateBase):
    observation_transformations: List[ObservationTransformationCreate] = None


class UpdateUpdate(UpdateBase):
    pass


class Update(UpdateBase):
    id: int

    class Config:
        orm_mode = True
