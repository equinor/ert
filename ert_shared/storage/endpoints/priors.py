from fastapi import APIRouter
from typing import List

from ert_shared.storage.db import Db, Session
from ert_shared.storage import json_schema as js, database_schema as ds


router = APIRouter()


@router.get("/priors", response_model=List[js.Prior])
async def read_responses(*, db: Session = Db(), ensemble_id: int):
    pass


@router.get("/ensembles/{ensemble_id}/responses/data")
async def read_responses_as_csv(*, db: Session = Db(), ensemble_id: int):
    pass


@router.get("/ensembles/{ensemble_id}/responses/{id}", response_model=js.Response)
async def read_response_by_id(*, db: Session = Db(), ensemble_id: int, id: int):
    return db.query(ds.Response).filter_by(id=id, ensemble_id=ensemble_id).one()


@router.get("/ensembles/{ensemble_id}/responses/{id}/data")
async def read_response_data(
    *,
    db: Session = Db(),
    ensemble_id: int,
    id: int,
):
    pass
