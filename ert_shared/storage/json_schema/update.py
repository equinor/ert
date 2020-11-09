from pydantic import BaseModel


class UpdateBase(BaseModel):
    algorithm: str


class UpdateCreate(UpdateBase):
    ensemble_name: str


class UpdateUpdate(UpdateBase):
    pass


class Update(UpdateBase):
    id: int
    ensemble_id: int

    class Config:
        orm_mode = True
