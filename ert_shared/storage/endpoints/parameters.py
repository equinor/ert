from typing import List
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import Response

from ert_shared.storage.db import Db, Session
from ert_shared.storage import json_schema as js, database_schema as ds


router = APIRouter()


@router.get("/ensembles/{ensemble_id}/parameters", response_model=List[js.Parameter])
async def parameters(*, db: Session = Db(), ensemble_id: int):
    return (
        db.query(ds.Parameter)
        .join(ds.Parameter.prior, isouter=True)
        .filter(ds.Parameter.ensemble_id == ensemble_id)
        .all()
    )


@router.post("/ensembles/{ensemble_id}/parameters", response_model=js.Parameter)
async def create_parameter(
    *,
    db: Session = Db(),
    ensemble_id: int,
):
    pass


@router.get("/ensembles/{ensemble_id}/parameters/{id}", response_model=js.Parameter)
async def parameter_by_id(*, db: Session = Db(), ensemble_id: int, id: int):
    return (
        db.query(ds.Parameter)
        .join(ds.Parameter.prior, isouter=True)
        .filter(
            ds.Parameter.id == id,
            ds.Parameter.ensemble_id == ensemble_id,
        )
        .one()
    )


@router.get("/ensembles/{ensemble_id}/parameters/{id}/data")
async def parameter_data_by_id(*, db: Session = Db(), ensemble_id: int, id: int):
    df = pd.DataFrame(
        [
            db.query(ds.Parameter.values)
            .filter_by(ensemble_id=ensemble_id, id=id)
            .one()
            .values
        ]
    ).T

    return Response(content=df.to_csv(index=True, header=False), media_type="text/csv")
