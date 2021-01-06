from typing import List, Mapping, Optional
from pydantic import BaseModel


class MisfitBase(BaseModel):
    pass


class MisfitCreate(MisfitBase):
    observation_key: str
    response_definition_key: str
    active: Optional[List[bool]] = None
    realizations: Mapping[int, float]


class MisfitUpdate(MisfitBase):
    pass


class Misfit(MisfitBase):
    id: int
    ensemble_id: Optional[int]
    name: str

    class Config:
        orm_mode = True
