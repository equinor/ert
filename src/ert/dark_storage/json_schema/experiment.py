from typing import Any, Mapping, Sequence
from uuid import UUID

from pydantic import BaseModel

from .prior import Prior


class _Experiment(BaseModel):
    name: str


class ExperimentIn(_Experiment):
    priors: Mapping[str, Prior] = {}


class ExperimentOut(_Experiment):
    id: UUID
    ensemble_ids: Sequence[UUID]
    priors: Mapping[str, Mapping[str, Any]]
    userdata: Mapping[str, Any]

    class Config:
        orm_mode = True
