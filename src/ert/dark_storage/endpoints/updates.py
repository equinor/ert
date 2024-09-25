from fastapi import APIRouter, Depends

from ert.dark_storage.enkf import get_storage

DEFAULT_STORAGE = Depends(get_storage)

router = APIRouter(tags=["ensemble"])


@router.post("/updates/facade")
def refresh_facade() -> None:
    pass
