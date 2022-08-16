from typing import Mapping, Any
from fastapi import APIRouter, Depends
from ert.shared.dark_storage.enkf import LibresFacade, get_res
from pathlib import Path

router = APIRouter(tags=["info"])


@router.get("/server/info", response_model=Mapping[str, Any])
def info(
    *,
    res: LibresFacade = Depends(get_res),
) -> Mapping[str, Any]:
    config = Path(res.user_config_file)
    return {"name": config.name}
