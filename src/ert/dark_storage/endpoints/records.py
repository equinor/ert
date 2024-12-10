import io
from collections.abc import Mapping
from typing import Annotated, Any
from urllib.parse import unquote
from uuid import UUID, uuid4

import numpy as np
from fastapi import APIRouter, Body, Depends, File, Header, HTTPException, status
from fastapi.responses import Response

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import (
    data_for_key,
    ensemble_parameters,
    gen_data_display_keys,
    get_observation_keys_for_response,
    get_observations_for_obs_keys,
    response_key_to_displayed_key,
)
from ert.dark_storage.enkf import get_storage
from ert.storage import Storage

router = APIRouter(tags=["record"])

DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)
DEFAULT_FILE = File(...)
DEFAULT_HEADER = Header("application/json")


@router.get("/ensembles/{ensemble_id}/records/{response_name}/observations")
async def get_record_observations(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
    response_name: str,
) -> list[js.ObservationOut]:
    response_name = unquote(response_name)
    ensemble = storage.get_ensemble(ensemble_id)
    obs_keys = get_observation_keys_for_response(ensemble, response_name)
    obss = get_observations_for_obs_keys(ensemble, obs_keys)

    obss.sort(key=lambda x: x["name"])
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


@router.get(
    "/ensembles/{ensemble_id}/records/{name}",
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {},
                "text/csv": {},
                "application/x-parquet": {},
            }
        },
        status.HTTP_401_UNAUTHORIZED: {
            "content": {
                "application/json": {"example": {"error": "Unauthorized access"}}
            },
        },
    },
)
async def get_ensemble_record(
    *,
    storage: Storage = DEFAULT_STORAGE,
    name: str,
    ensemble_id: UUID,
    accept: Annotated[str | None, Header()] = None,
) -> Any:
    name = unquote(name)
    try:
        dataframe = data_for_key(storage.get_ensemble(ensemble_id), name)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    media_type = accept if accept is not None else "text/csv"
    if media_type == "application/x-parquet":
        dataframe.columns = [str(s) for s in dataframe.columns]
        stream = io.BytesIO()
        dataframe.to_parquet(stream)
        return Response(
            content=stream.getvalue(),
            media_type="application/x-parquet",
        )
    elif media_type == "application/json":
        return Response(dataframe.to_json(), media_type="application/json")
    else:
        return Response(
            content=dataframe.to_csv().encode(),
            media_type="text/csv",
        )


@router.get("/ensembles/{ensemble_id}/parameters", response_model=list[dict[str, Any]])
async def get_ensemble_parameters(
    *, storage: Storage = DEFAULT_STORAGE, ensemble_id: UUID
) -> list[dict[str, Any]]:
    return ensemble_parameters(storage, ensemble_id)


@router.get(
    "/ensembles/{ensemble_id}/responses", response_model=Mapping[str, js.RecordOut]
)
def get_ensemble_responses(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
) -> Mapping[str, js.RecordOut]:
    response_map: dict[str, js.RecordOut] = {}
    ensemble = storage.get_ensemble(ensemble_id)

    response_names_with_observations = set()
    observations = ensemble.experiment.observations

    if len(ensemble.has_data()) == 0:
        return {}

    for (
        response_type,
        response_config,
    ) in ensemble.experiment.response_configuration.items():
        if response_type in observations:
            obs_ds = observations[response_type]
            display_key_fn = response_key_to_displayed_key[response_type]
            obs_with_responses = (
                obs_ds.select(["response_key", *response_config.primary_key])
                .map_rows(display_key_fn)
                .unique()
                .to_series()
                .to_list()
            )
            response_names_with_observations.update(set(obs_with_responses))

    for name in ensemble.experiment.response_type_to_response_keys.get("summary", []):
        response_map[str(name)] = js.RecordOut(
            id=UUID(int=0),
            name=name,
            userdata={"data_origin": "Summary"},
            has_observations=name in response_names_with_observations,
        )

    if "gen_data" in ensemble.experiment.response_type_to_response_keys:
        for name in gen_data_display_keys(ensemble):
            response_map[str(name)] = js.RecordOut(
                id=UUID(int=0),
                name=name,
                userdata={"data_origin": "GEN_DATA"},
                has_observations=name in response_names_with_observations,
            )

    return response_map


@router.get("/ensembles/{ensemble_id}/records/{key}/std_dev")
def get_std_dev(
    *, storage: Storage = DEFAULT_STORAGE, ensemble_id: UUID, key: str, z: int
) -> Response:
    key = unquote(key)
    ensemble = storage.get_ensemble(ensemble_id)
    try:
        da = ensemble.calculate_std_dev_for_parameter(key)["values"]
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Data not found") from e

    if z >= int(da.shape[2]):
        raise HTTPException(status_code=400, detail="Invalid z index")

    data_2d = da[:, :, z]

    buffer = io.BytesIO()
    np.save(buffer, data_2d)

    return Response(content=buffer.getvalue(), media_type="application/octet-stream")
