from typing import List, Optional
from pydantic import BaseModel


class PriorBase(BaseModel):
    group: str
    key: str
    function: str
    parameter_names: List[str]
    parameter_values: List[float]


class PriorCreate(PriorBase):
    pass


class Prior(PriorBase):
    id: Optional[int] = None

    class Config:
        orm_mode = True
