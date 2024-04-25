from typing import Any, List, Mapping, Optional
from uuid import UUID

from pydantic import BaseModel


class _Ensemble(BaseModel):
    size: int
    active_realizations: List[int] = []


class EnsembleIn(_Ensemble):
    update_id: Optional[UUID] = None
    userdata: Mapping[str, Any] = {}


class EnsembleOut(_Ensemble):
    id: UUID
    experiment_id: Optional[UUID] = None
    userdata: Mapping[str, Any]
