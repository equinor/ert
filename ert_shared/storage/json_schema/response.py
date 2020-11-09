from typing import Any, List, Mapping, Optional
from pydantic import BaseModel


class ResponseBase(BaseModel):
    pass


class ResponseCreate(ResponseBase):
    realizations: Mapping[int, List[float]]
    name: str
    indices: List[Any]


class ResponseUpdate(ResponseBase):
    pass


class Response(ResponseBase):
    id: int
    ensemble_id: Optional[int]
    name: str

    class Config:
        orm_mode = True
