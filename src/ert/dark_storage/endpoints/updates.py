from uuid import UUID

from ert_storage import json_schema as js
from fastapi import APIRouter, Depends

from ert.dark_storage.enkf import LibresFacade, get_res, reset_res

router = APIRouter(tags=["ensemble"])


@router.post("/updates", response_model=js.UpdateOut)
def create_update(
    *,
    res: LibresFacade = Depends(get_res),
    update: js.UpdateIn,
) -> js.UpdateOut:
    raise NotImplementedError


@router.get("/updates/{update_id}", response_model=js.UpdateOut)
def get_update(
    *,
    res: LibresFacade = Depends(get_res),
    update_id: UUID,
) -> js.UpdateOut:
    raise NotImplementedError


@router.post("/updates/facade")
def refresh_facade(
    *,
    res: LibresFacade = Depends(reset_res),
) -> None:
    if res is not None:
        raise ValueError("Could not clean the ert facade")
