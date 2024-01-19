from fastapi import APIRouter, Depends

from ert.dark_storage.enkf import LibresFacade, get_res, reset_res

router = APIRouter(tags=["ensemble"])

DEFAULT_LIBRESFACADE = Depends(get_res)
LIBRESFACADE_RESET_RES = Depends(reset_res)


@router.post("/updates/facade")
def refresh_facade(*, res: LibresFacade = LIBRESFACADE_RESET_RES) -> None:
    if res is not None:
        raise ValueError("Could not clean the ert facade")
