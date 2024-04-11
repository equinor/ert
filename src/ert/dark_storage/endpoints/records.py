import io
from typing import Any, Dict, List, Mapping, Union
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, File, Header, status
from fastapi.responses import Response
from typing_extensions import Annotated

from ert.dark_storage import json_schema as js
from ert.dark_storage.common import (
    data_for_key,
    ensemble_parameters,
    gen_data_keys,
    get_observation_for_response,
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
) -> List[js.ObservationOut]:
    ensemble = storage.get_ensemble(ensemble_id)
    obs = get_observation_for_response(ensemble, response_name)

    if obs is None:
        return []

    return [
        js.ObservationOut(
            id=uuid4(),
            userdata={},
            errors=obs.errors,
            values=obs.values,
            x_axis=obs.x_axis,
            name=obs.obs_name,
        )
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
        }
    },
)
async def get_ensemble_record(
    *,
    storage: Storage = DEFAULT_STORAGE,
    name: str,
    ensemble_id: UUID,
    accept: Annotated[Union[str, None], Header()] = None,
) -> Any:
    dataframe = data_for_key(storage.get_ensemble(ensemble_id), name)

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


@router.get("/ensembles/{ensemble_id}/parameters", response_model=List[Dict[str, Any]])
async def get_ensemble_parameters(
    *, storage: Storage = DEFAULT_STORAGE, ensemble_id: UUID
) -> List[Dict[str, Any]]:
    return ensemble_parameters(storage, ensemble_id)


@router.get(
    "/ensembles/{ensemble_id}/responses", response_model=Mapping[str, js.RecordOut]
)
def get_ensemble_responses(
    *,
    storage: Storage = DEFAULT_STORAGE,
    ensemble_id: UUID,
) -> Mapping[str, js.RecordOut]:
    response_map: Dict[str, js.RecordOut] = {}
    ensemble = storage.get_ensemble(ensemble_id)

    response_names_with_observations = set()
    observations = ensemble.experiment.observations

    if "gen_data" in observations:
        gen_obs_ds = observations["gen_data"]
        response_names_with_observations.update(
            [
                f"{x}@{','.join(map(str, gen_obs_ds.sel(name=x).report_step.values.flatten()))}"
                for x in gen_obs_ds["name"].data
            ]
        )

        for name in gen_data_keys(ensemble):
            response_map[str(name)] = js.RecordOut(
                id=UUID(int=0),
                name=name,
                userdata={"data_origin": "GEN_DATA"},
                has_observations=name in response_names_with_observations,
            )

    if "summary" in observations:
        summary_ds = observations["summary"]
        response_names_with_observations.update(summary_ds["name"].data)

        for name in ensemble.get_summary_keyset():
            response_map[str(name)] = js.RecordOut(
                id=UUID(int=0),
                name=name,
                userdata={"data_origin": "Summary"},
                has_observations=name in response_names_with_observations,
            )

    return response_map
