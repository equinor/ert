from fastapi import APIRouter

from ert_shared.storage.db import Db, Session
from ert_shared.storage import json_schema as js, database_schema as ds


router = APIRouter()


@router.get("/ensembles/{ensemble_id}/realizations/{index}")
async def read_item(*, db: Session = Db(), ensemble_id: int, index: int):
    realization = (
        db.query(ds.Realization).filter_by(ensemble_id=ensemble_id, index=index).one()
    )

    response_definitions = db.query(ds.ResponseDefinition).filter_by(
        ensemble_id=ensemble_id
    )
    responses = [
        {
            "name": resp_def.name,
            "response": (
                db.query(ds.Response)
                .filter_by(
                    response_definition_id=resp_def.id, realization_id=realization.id
                )
                .one()
            ),
        }
        for resp_def in response_definitions
    ]

    parameter_definitions = db.query(ds.ParameterDefinition).filter_by(
        ensemble_id=ensemble_id
    )
    parameters = [
        {
            "name": param_def.name,
            "parameter": (
                db.query(ds.Parameter)
                .filter_by(
                    parameter_definition_id=param_def.id, realization_id=realization.id
                )
                .one()
            ),
        }
        for param_def in parameter_definitions
    ]

    return_schema = {
        "name": realization.index,
        "responses": [
            {"name": res["name"], "data": res["response"].values} for res in responses
        ],
        "parameters": [
            {"name": par["name"], "data": par["parameter"].value} for par in parameters
        ],
    }

    return return_schema
