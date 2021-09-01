from pathlib import Path
from typing import Optional
from fastapi import Depends
from ert_storage.security import security
from ert_shared.libres_facade import LibresFacade
from ert_shared.ert_adapter import ERT
from res.enkf import EnKFMain, ResConfig


__all__ = ["LibresFacade", "get_res"]


_libres_facade: Optional[LibresFacade] = None


def init_facade() -> None:
    global _libres_facade

    # Assuming this project is installed in development mode from a git
    # repository.
    thisdir = Path(__file__).parent
    gitdir = thisdir.parent.parent # ert_shared/dark_storage
    configfile = gitdir / "test-data" / "local" / "poly_example" / "poly.ert"

    config = ResConfig(str(configfile))
    ert = EnKFMain(config)
    _libres_facade = LibresFacade(ert)


async def get_res(*, _: None = Depends(security)) -> LibresFacade:
    global _libres_facade
    if _libres_facade is None:
        raise ValueError("LibresFacade is not initialised")
    return _libres_facade
