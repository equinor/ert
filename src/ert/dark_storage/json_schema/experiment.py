from typing import Any, Dict, List, Mapping
from uuid import UUID

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from .prior import Prior


@dataclass
class _Experiment:
    name: str


@dataclass
class ExperimentIn(_Experiment):
    priors: Mapping[str, Prior] = Field(default_factory=dict)


@dataclass(config=ConfigDict(from_attributes=True))
class ExperimentOut(_Experiment):
    id: UUID
    ensemble_ids: List[UUID]
    priors: Mapping[str, Dict[str, Any]]
    userdata: Mapping[str, Any]
