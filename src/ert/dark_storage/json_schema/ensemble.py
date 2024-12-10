from collections.abc import Mapping
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class _Ensemble(BaseModel):
    size: int
    active_realizations: list[int] = []


class EnsembleIn(_Ensemble):
    update_id: UUID | None = None
    userdata: Mapping[str, Any] = {}


class EnsembleOut(_Ensemble):
    id: UUID
    experiment_id: UUID | None = None
    userdata: Mapping[str, Any]
