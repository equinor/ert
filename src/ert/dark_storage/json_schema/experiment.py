from collections.abc import Mapping
from typing import Any
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
    ensemble_ids: list[UUID]
    priors: Mapping[str, dict[str, Any]]
    userdata: Mapping[str, Any]
    parameters: Mapping[str, list[dict[str, Any]]]
    responses: Mapping[str, list[dict[str, Any]]]
    observations: Mapping[str, dict[str, list[str]]]
