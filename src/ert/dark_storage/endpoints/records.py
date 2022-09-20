import io
from typing import Any, List, Mapping, Optional
from uuid import UUID, uuid4

import pandas as pd
from ert_storage import json_schema as js
from fastapi import APIRouter, Body, Depends, File, Header, Request, UploadFile, status
from fastapi.responses import Response

from ert.dark_storage.common import (
    data_for_key,
    ensemble_parameters,
    observations_for_obs_keys,
)
from ert.dark_storage.enkf import LibresFacade, get_id, get_name, get_res

router = APIRouter(tags=["record"])


@router.post("/ensembles/{ensemble_id}/records/{name}/file")
async def post_ensemble_record_file(
    *,
    res: LibresFacade = Depends(get_res),
    name: str,
    ensemble_id: UUID,
    realization_index: Optional[int] = None,
    file: UploadFile = File(...),
) -> None:
    raise NotImplementedError


@router.put("/ensembles/{ensemble_id}/records/{name}/blob")
async def add_block(
    *,
    res: LibresFacade = Depends(get_res),
    name: str,
    ensemble_id: UUID,
    block_index: int,
    realization_index: Optional[int] = None,
    request: Request,
) -> None:
    raise NotImplementedError


@router.post("/ensembles/{ensemble_id}/records/{name}/blob")
async def create_blob(
    *,
    res: LibresFacade = Depends(get_res),
    name: str,
    ensemble_id: UUID,
    realization_index: Optional[int] = None,
) -> None:
    raise NotImplementedError


@router.patch("/ensembles/{ensemble_id}/records/{name}/blob")
async def finalize_blob(
    *,
    res: LibresFacade = Depends(get_res),
    name: str,
    ensemble_id: UUID,
    realization_index: Optional[int] = None,
) -> None:
    raise NotImplementedError


@router.post(
    "/ensembles/{ensemble_id}/records/{name}/matrix", response_model=js.RecordOut
)
async def post_ensemble_record_matrix(
    *,
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
    name: str,
    prior: Optional[str] = None,
    realization_index: Optional[int] = None,
    content_type: str = Header("application/json"),
    request: Request,
) -> js.RecordOut:
    raise NotImplementedError


@router.put("/ensembles/{ensemble_id}/records/{name}/userdata")
async def replace_record_userdata(
    *,
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
    name: str,
    realization_index: Optional[int] = None,
    body: Any = Body(...),
) -> None:
    raise NotImplementedError


@router.patch("/ensembles/{ensemble_id}/records/{name}/userdata")
async def patch_record_userdata(
    *,
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
    name: str,
    realization_index: Optional[int] = None,
    body: Any = Body(...),
) -> None:
    raise NotImplementedError


@router.get(
    "/ensembles/{ensemble_id}/records/{name}/userdata", response_model=Mapping[str, Any]
)
async def get_record_userdata(
    *,
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
    name: str,
    realization_index: Optional[int] = None,
) -> Mapping[str, Any]:
    raise NotImplementedError


@router.post("/ensembles/{ensemble_id}/records/{name}/observations")
async def post_record_observations(
    *,
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
    name: str,
    realization_index: Optional[int] = None,
    observation_ids: List[UUID] = Body(...),
) -> None:
    raise NotImplementedError


@router.get("/ensembles/{ensemble_id}/records/{name}/observations")
async def get_record_observations(
    *,
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
    name: str,
    realization_index: Optional[int] = None,
) -> List[js.ObservationOut]:
    obs_keys = res.observation_keys(name)
    case = get_name("ensemble", ensemble_id)
    obss = observations_for_obs_keys(res, case, obs_keys)
    return [
        js.ObservationOut(
            id=uuid4(),
            userData=[],
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
                "application/x-numpy": {},
            }
        }
    },
)
async def get_ensemble_record(
    *,
    res: LibresFacade = Depends(get_res),
    name: str,
    ensemble_id: UUID,
    accept: str = Header("application/json"),
    realization_index: Optional[int] = None,
    label: Optional[str] = None,
) -> Any:
    ensemble_name = get_name("ensemble", ensemble_id)
    dataframe = data_for_key(res, ensemble_name, name, realization_index)
    if realization_index is not None:
        # dataframe.loc returns a Series, and when we reconstruct a DataFrame
        # from a Series, it defaults to be oriented the wrong way, so we must
        # transpose it
        dataframe = pd.DataFrame(dataframe.loc[realization_index]).T

    if accept == "application/x-parquet":
        dataframe.columns = [str(s) for s in dataframe.columns]
        stream = io.BytesIO()
        dataframe.to_parquet(stream)
        return Response(
            content=stream.getvalue(),
            media_type=accept,
        )
    elif accept == "application/json":
        return Response(dataframe.to_json(), media_type=accept)
    else:
        return Response(
            content=dataframe.to_csv().encode(),
            media_type="text/csv",
        )


@router.get("/ensembles/{ensemble_id}/records/{name}/labels", response_model=List[str])
async def get_record_labels(
    *,
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
    name: str,
) -> List[str]:
    return []


@router.get("/ensembles/{ensemble_id}/parameters", response_model=List[dict])
async def get_ensemble_parameters(
    *, res: LibresFacade = Depends(get_res), ensemble_id: UUID
) -> List[dict]:
    return ensemble_parameters(res)


@router.get(
    "/ensembles/{ensemble_id}/records", response_model=Mapping[str, js.RecordOut]
)
async def get_ensemble_records(
    *, res: LibresFacade = Depends(get_res), ensemble_id: UUID
) -> Mapping[str, js.RecordOut]:
    raise NotImplementedError


@router.get("/records/{record_id}", response_model=js.RecordOut)
async def get_record(
    *, res: LibresFacade = Depends(get_res), record_id: UUID
) -> js.RecordOut:
    raise NotImplementedError


@router.get("/records/{record_id}/data")
async def get_record_data(
    *,
    res: LibresFacade = Depends(get_res),
    record_id: UUID,
    accept: Optional[str] = Header(default="application/json"),
) -> Any:
    raise NotImplementedError


@router.get(
    "/ensembles/{ensemble_id}/responses", response_model=Mapping[str, js.RecordOut]
)
def get_ensemble_responses(
    *, res: LibresFacade = Depends(get_res), ensemble_id: UUID
) -> Mapping[str, js.RecordOut]:

    response_map: Mapping[str, js.RecordOut] = {}

    for name in res.get_summary_keys():
        obs_keys = res.observation_keys(name)
        response_map[str(name)] = js.RecordOut(
            id=get_id("response", f"{ensemble_id}/{name}"),
            name=name,
            userdata={"data_origin": "Summary"},
            has_observations=len(obs_keys) != 0,
        )

    for name in res.get_gen_data_keys():
        obs_keys = res.observation_keys(name)
        response_map[str(name)] = js.RecordOut(
            id=get_id("response", f"{ensemble_id}/{name}"),
            name=name,
            userdata={"data_origin": "GEN_DATA"},
            has_observations=len(obs_keys) != 0,
        )

    return response_map
