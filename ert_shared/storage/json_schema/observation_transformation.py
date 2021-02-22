from typing import List, Union
from pydantic import BaseModel
from datetime import datetime


class ObservationTransformationBase(BaseModel):
    name: str
    active: List[bool]
    scale: List[int]


class ObservationTransformationCreate(ObservationTransformationBase):
    pass


class ObservationTransformation(ObservationTransformationBase):
    id: int

    class Config:
        orm_mode = True
