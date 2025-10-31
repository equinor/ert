import logging

from fastapi import APIRouter, Body, Depends

from ert.dark_storage.common import get_storage, get_storage_api_version

router = APIRouter(tags=["version"])
logger = logging.getLogger(__name__)

DEFAULT_STORAGE = Depends(get_storage)
DEFAULT_BODY = Body(...)


@router.get("/version")
def get_version() -> str:
    return get_storage_api_version()
