from pathlib import Path
from typing import Any, Mapping

from fastapi import APIRouter, Depends

from ert.dark_storage.enkf import LibresFacade, get_res

router = APIRouter(tags=["info"])


@router.get("/server/info", response_model=Mapping[str, Any])
def info(
    *,
    res: LibresFacade = Depends(get_res),
) -> Mapping[str, Any]:
    config = Path(res.user_config_file)
    return {"name": config.name}
