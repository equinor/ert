from typing import Any, Mapping
from uuid import UUID

from ert_storage import json_schema as js
from fastapi import APIRouter, Body, Depends

from ert.dark_storage.common import ensemble_parameter_names, get_response_names
from ert.dark_storage.enkf import LibresFacade, get_id, get_name, get_res, get_size

router = APIRouter(tags=["ensemble"])


@router.post("/experiments/{experiment_id}/ensembles", response_model=js.EnsembleOut)
def post_ensemble(
    *, res: LibresFacade = Depends(get_res), ens_in: js.EnsembleIn, experiment_id: UUID
) -> js.EnsembleOut:
    raise NotImplementedError


@router.get("/ensembles/{ensemble_id}", response_model=js.EnsembleOut)
def get_ensemble(
    *, res: LibresFacade = Depends(get_res), ensemble_id: UUID
) -> js.EnsembleOut:
    return js.EnsembleOut(
        id=ensemble_id,
        children=[],
        parent=None,
        experiment_id=get_id("experiment", "default"),
        userdata={"name": get_name("ensemble", ensemble_id)},
        size=get_size(res),
        parameter_names=ensemble_parameter_names(res),
        response_names=get_response_names(res, get_name("ensemble", ensemble_id)),
        child_ensemble_ids=[],
    )


@router.put("/ensembles/{ensemble_id}/userdata")
async def replace_ensemble_userdata(
    *,
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
    body: Any = Body(...),
) -> None:
    raise NotImplementedError


@router.patch("/ensembles/{ensemble_id}/userdata")
async def patch_ensemble_userdata(
    *,
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
    body: Any = Body(...),
) -> None:
    raise NotImplementedError


@router.get("/ensembles/{ensemble_id}/userdata", response_model=Mapping[str, Any])
async def get_ensemble_userdata(
    *,
    res: LibresFacade = Depends(get_res),
    ensemble_id: UUID,
) -> Mapping[str, Any]:
    raise NotImplementedError
