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
        db.query(ds.ParameterDefinition)
        .join(ds.ParameterDefinition.prior, isouter=True)
        .filter(ds.ParameterDefinition.ensemble_id == ensemble_id)
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
        db.query(ds.ParameterDefinition)
        .join(ds.ParameterDefinition.prior, isouter=True)
        .filter(
            ds.ParameterDefinition.id == id,
            ds.ParameterDefinition.ensemble_id == ensemble_id,
        )
        .one()
    )


@router.get("/ensembles/{ensemble_id}/parameters/{id}/data")
async def parameter_data_by_id(*, db: Session = Db(), ensemble_id: int, id: int):
    (
        db.query(ds.ParameterDefinition)
        .filter(
            ds.ParameterDefinition.ensemble_id == ensemble_id,
            ds.ParameterDefinition.id == id,
        )
        .one()
    )

    df = pd.DataFrame(
        [
            [ref.realization.index, ref.value]
            for ref in (
                db.query(ds.Parameter)
                .filter(ds.Parameter.parameter_definition_id == id)
                .all()
            )
        ]
    )

    return Response(content=df.to_csv(index=False, header=False), media_type="text/csv")
