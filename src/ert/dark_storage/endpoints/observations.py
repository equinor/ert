from typing import List
from uuid import UUID

from fastapi import APIRouter, Body, Depends

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import get_all_observations
from ert.dark_storage.enkf import get_storage
from ert.storage import Storage

router = APIRouter(tags=["ensemble"])

DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


@router.get(
    "/experiments/{experiment_id}/observations", response_model=List[js.ObservationOut]
)
def get_observations(
    *, storage: Storage = DEFAULT_STORAGE, experiment_id: UUID
) -> List[js.ObservationOut]:
    experiment = storage.get_experiment(experiment_id)
    return [
        js.ObservationOut(
            id=UUID(int=0),
            userdata={},
            errors=observation["errors"],
            values=observation["values"],
            x_axis=observation["x_axis"],
            name=observation["name"],
        )
        for observation in get_all_observations(experiment)
    ]
