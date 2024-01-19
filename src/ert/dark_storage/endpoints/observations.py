from typing import List
from uuid import UUID

from fastapi import APIRouter, Body, Depends

from ert.dark_storage import json_schema as js
from ert.dark_storage.enkf import LibresFacade, get_res
from ert.shared.storage.extraction import create_observations

router = APIRouter(tags=["ensemble"])

DEFAULT_LIBRESFACADE = Depends(get_res)
DEFAULT_BODY = Body(...)


@router.get(
    "/experiments/{experiment_id}/observations", response_model=List[js.ObservationOut]
)
def get_observations(
    *, res: LibresFacade = DEFAULT_LIBRESFACADE, experiment_id: UUID
) -> List[js.ObservationOut]:
    return [
        js.ObservationOut(
            id=UUID(int=0),
            userdata={},
            errors=obs["errors"],
            values=obs["values"],
            x_axis=obs["x_axis"],
            name=obs["name"],
        )
        for obs in create_observations(res)
    ]
