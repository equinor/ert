import operator
from urllib.parse import unquote
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import (
    get_all_observations,
    get_observations_for_obs_keys,
    get_storage,
)
from ert.storage import Storage

router = APIRouter(tags=["ensemble"])

DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


@router.get(
    "/experiments/{experiment_id}/observations", response_model=list[js.ObservationOut]
)
def get_observations(
    *, storage: Storage = DEFAULT_STORAGE, experiment_id: UUID
) -> list[js.ObservationOut]:
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


@router.get("/ensembles/{ensemble_id}/responses/{response_key}/observations")
async def get_observations_for_response(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
    response_key: str,
) -> list[js.ObservationOut]:
    response_key = unquote(response_key)
    ensemble = storage.get_ensemble(ensemble_id)
    experiment = ensemble.experiment

    response_type = experiment.response_key_to_response_type.get(response_key, "")
    obs_keys = experiment.response_key_to_observation_key.get(response_type, {}).get(
        response_key
    )
    if not obs_keys:
        return []

    obss = get_observations_for_obs_keys(ensemble, obs_keys)

    obss.sort(key=operator.itemgetter("name"))
    if not obss:
        return []

    return [
        js.ObservationOut(
            id=uuid4(),
            userdata={},
            errors=obs["errors"],
            values=obs["values"],
            x_axis=obs["x_axis"],
            name=obs["name"],
        )
        for obs in obss
    ]
