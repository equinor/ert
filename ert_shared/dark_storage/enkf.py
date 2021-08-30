from fastapi import Depends
from ert_storage.security import security
from ert_shared.libres_facade import LibresFacade
from ert_shared.ert_adapter import ERT


__all__ = ["LibresFacade", "get_res"]


async def get_res(*, _: None = Depends(security)) -> LibresFacade:
    return ERT.enkf_facade
