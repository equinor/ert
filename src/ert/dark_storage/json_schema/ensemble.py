from typing import Any, List, Mapping, Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class _Ensemble(BaseModel):
    size: int
    parameter_names: List[str]
    response_names: List[str]
    active_realizations: List[int] = []


class EnsembleIn(_Ensemble):
    update_id: Optional[UUID] = None
    userdata: Mapping[str, Any] = {}

    @model_validator(mode="after")
    def check_names_no_overlap(self) -> Self:
        """
        Verify that `parameter_names` and `response_names` don't overlap. Ie, no
        record can be both a parameter and a response.
        """
        if not set(self.parameter_names).isdisjoint(set(self.response_names)):
            raise ValueError("parameters and responses cannot have a name in common")
        return self


class EnsembleOut(_Ensemble):
    id: UUID
    children: List[UUID] = Field(alias="child_ensemble_ids")
    parent: Optional[UUID] = Field(None, alias="parent_ensemble_id")
    experiment_id: Optional[UUID] = None
    userdata: Mapping[str, Any]

    class Config:
        from_attributes = True
