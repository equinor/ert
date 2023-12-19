from typing import Any, Dict, List, Mapping
from uuid import UUID

from pydantic import BaseModel

from .prior import Prior


class _Experiment(BaseModel):
    name: str


class ExperimentIn(_Experiment):
    priors: Mapping[str, Prior] = {}


class ExperimentOut(_Experiment):
    id: UUID
    ensemble_ids: List[UUID]
    priors: Mapping[str, Dict[str, Any]]
    userdata: Mapping[str, Any]

    class Config:
        from_attributes = True
