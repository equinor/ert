from pathlib import Path
from typing import Any, Mapping

from fastapi import APIRouter, Depends

from ert.dark_storage.enkf import LibresFacade, get_res

router = APIRouter(tags=["info"])
DEFAULT_LIBRESFACADE = Depends(get_res)


@router.get("/server/info", response_model=Mapping[str, Any])
def info(
    *,
    res: LibresFacade = DEFAULT_LIBRESFACADE,
) -> Mapping[str, Any]:
    config = Path(res.user_config_file)  # type: ignore
    return {"name": config.name}
