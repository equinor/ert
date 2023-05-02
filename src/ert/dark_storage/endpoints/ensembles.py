from typing import Any, Mapping
from uuid import UUID

from ert_storage import json_schema as js
from fastapi import APIRouter, Body, Depends

from ert.dark_storage.common import ensemble_parameter_names, get_response_names
from ert.dark_storage.enkf import LibresFacade, get_res, get_storage
from ert.storage import StorageAccessor

router = APIRouter(tags=["ensemble"])
DEFAULT_LIBRESFACADE = Depends(get_res)
DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


@router.post("/experiments/{experiment_id}/ensembles", response_model=js.EnsembleOut)
def post_ensemble(
    *,
    res: LibresFacade = DEFAULT_LIBRESFACADE,
    ens_in: js.EnsembleIn,
    experiment_id: UUID,
) -> js.EnsembleOut:
    raise NotImplementedError


@router.get("/ensembles/{ensemble_id}", response_model=js.EnsembleOut)
def get_ensemble(
    *,
    res: LibresFacade = DEFAULT_LIBRESFACADE,
    db: StorageAccessor = DEFAULT_STORAGE,
    ensemble_id: UUID,
) -> js.EnsembleOut:
    ens = db.get_ensemble(ensemble_id)
    return js.EnsembleOut(
        id=ensemble_id,
        children=[],
        parent=None,
        experiment_id=ens.experiment_id,
        userdata={"name": ens.name},
        size=ens.ensemble_size,
        parameter_names=ensemble_parameter_names(res),
        response_names=get_response_names(res, ens),
        child_ensemble_ids=[],
    )


@router.put("/ensembles/{ensemble_id}/userdata")
async def replace_ensemble_userdata(
    *,
    res: LibresFacade = DEFAULT_LIBRESFACADE,
    ensemble_id: UUID,
    body: Any = DEFAULT_BODY,
) -> None:
    raise NotImplementedError


@router.patch("/ensembles/{ensemble_id}/userdata")
async def patch_ensemble_userdata(
    *,
    res: LibresFacade = DEFAULT_LIBRESFACADE,
    ensemble_id: UUID,
    body: Any = DEFAULT_BODY,
) -> None:
    raise NotImplementedError


@router.get("/ensembles/{ensemble_id}/userdata", response_model=Mapping[str, Any])
async def get_ensemble_userdata(
    *,
    res: LibresFacade = DEFAULT_LIBRESFACADE,
    ensemble_id: UUID,
) -> Mapping[str, Any]:
    raise NotImplementedError
