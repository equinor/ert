from uuid import UUID
from typing import List, Mapping, Any
from pydantic import BaseModel
from .prior import Prior


class _Experiment(BaseModel):
    name: str


class ExperimentIn(_Experiment):
    priors: Mapping[str, Prior] = {}


class ExperimentOut(_Experiment):
    id: UUID
    ensembles: List[UUID]
    priors: Mapping[str, UUID]
    metadata: Mapping[str, Any] = {}

    class Config:
        orm_mode = True
