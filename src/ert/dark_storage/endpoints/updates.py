from uuid import UUID

from ert_storage import json_schema as js
from fastapi import APIRouter, Depends

from ert.dark_storage.enkf import LibresFacade, get_res, reset_res

router = APIRouter(tags=["ensemble"])

DEFAULT_LIBRESFACADE = Depends(get_res)
LIBRESFACADE_RESET_RES = Depends(reset_res)


@router.post("/updates", response_model=js.UpdateOut)
def create_update(
    *,
    res: LibresFacade = DEFAULT_LIBRESFACADE,
    update: js.UpdateIn,
) -> js.UpdateOut:
    raise NotImplementedError


@router.get("/updates/{update_id}", response_model=js.UpdateOut)
def get_update(
    *,
    res: LibresFacade = DEFAULT_LIBRESFACADE,
    update_id: UUID,
) -> js.UpdateOut:
    raise NotImplementedError


@router.post("/updates/facade")
def refresh_facade(*, res: LibresFacade = LIBRESFACADE_RESET_RES) -> None:
    if res is not None:
        raise ValueError("Could not clean the ert facade")
