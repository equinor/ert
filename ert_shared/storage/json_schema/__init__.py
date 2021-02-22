from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

from .prior import Prior, PriorCreate
from .ensemble import Ensemble, EnsembleCreate
from .response import Response, ResponseCreate
from .parameter import Parameter, ParameterCreate
from .observation import Observation, ObservationCreate
from .observation_transformation import (
    ObservationTransformation,
    ObservationTransformationCreate,
)
from .misfit import Misfit, MisfitCreate
from .update import Update, UpdateCreate


__all__ = [
    "Ensemble",
    "EnsembleCreate",
    "Healthcheck",
    "Misfit",
    "MisfitCreate",
    "Observation",
    "ObservationCreate",
    "ObservationTransformation",
    "ObservationTransformationCreate",
    "Parameter",
    "ParameterCreate",
    "Prior",
    "PriorCreate",
    "Project",
    "Realization",
    "Response",
    "ResponseCreate",
    "Update",
    "UpdateCreate",
]


class Healthcheck(BaseModel):
    """Used exclusively by the /healthcheck endpoint"""

    date: datetime


class Project(BaseModel):
    ensembles: Optional[List[Ensemble]]
    observations: Optional[List[Observation]]


class Realization(BaseModel):
    class Config:
        orm_mode = True

    id: int
    ensemble_id: int
