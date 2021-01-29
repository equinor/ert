from fastapi import APIRouter

from ert_shared.storage.db import Db, Session
from ert_shared.storage import json_schema as js, database_schema as ds


router = APIRouter()


@router.get("/ensembles/{ensemble_id}/realizations/{index}")
async def read_item(*, db: Session = Db(), ensemble_id: int, index: int):
    response_definitions = db.query(ds.ResponseDefinition).filter_by(
        ensemble_id=ensemble_id
    )
    responses = [
        {
            "name": resp_def.name,
            "response": (
                db.query(ds.Response)
                .filter_by(response_definition_id=resp_def.id, index=index)
                .one()
            ),
        }
        for resp_def in response_definitions
    ]

    parameters = [
        {
            "name": param.name,
            "value": param.values[index],
        }
        for param in db.query(ds.Parameter).filter_by(ensemble_id=ensemble_id)
    ]

    return_schema = {
        "name": index,
        "responses": [
            {"name": res["name"], "data": res["response"].values} for res in responses
        ],
        "parameters": [
            {"name": par["name"], "data": par["value"]} for par in parameters
        ],
    }

    return return_schema
