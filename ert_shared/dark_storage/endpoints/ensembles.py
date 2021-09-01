from uuid import UUID
from typing import Any, Mapping

from fastapi import APIRouter, Body, Depends
from ert_storage import json_schema as js
from ert_shared.dark_storage.enkf import LibresFacade, get_res

router = APIRouter(tags=["ensemble"])


@router.post("/experiments/{experiment_id}/ensembles", response_model=js.EnsembleOut)
def post_ensemble(
    *, res: LibresFacade = Depends(get_res), ens_in: js.EnsembleIn, experiment_id: UUID
) -> js.EnsembleOut:
    raise NotImplementedError


@router.get("/ensembles/{ensemble_id}", response_model=js.EnsembleOut)
def get_ensemble(
    *, res: LibresFacade = Depends(get_res), ensemble_id: int, zohar: str
) -> js.EnsembleOut:
    raise NotImplementedError


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
