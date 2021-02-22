from typing import Any, List, Optional, Mapping
from pydantic import BaseModel
from datetime import datetime

from ert_shared.storage.json_schema.parameter import ParameterCreate
from ert_shared.storage.json_schema.prior import PriorCreate
from ert_shared.storage.json_schema.update import Update
from ert_shared.storage.json_schema.response import Response
from ert_shared.storage.json_schema.observation import ObservationCreate


class EnsembleBase(BaseModel):
    name: str


class EnsembleCreate(EnsembleBase):
    parameters: List[ParameterCreate]
    priors: List[PriorCreate]
    realizations: int
    update_id: Optional[int] = None
    observations: List[ObservationCreate] = None
    response_observation_link: Mapping[str, str]


class EnsembleUpdate(EnsembleBase):
    pass


class Ensemble(EnsembleBase):
    id: Optional[int] = None
    time_created: datetime
    children: List[Any]
    parent: Optional[Any] = None
    responses: Optional[List[Response]] = None

    class Config:
        orm_mode = True
