from __future__ import annotations

import os
from typing import Optional

from ert_storage.security import security
from fastapi import Depends

from ert.config import ErtConfig
from ert.enkf_main import EnKFMain
from ert.libres_facade import LibresFacade
from ert.storage import StorageReader, open_storage

__all__ = ["LibresFacade", "get_res", "get_storage"]


_libres_facade: Optional[LibresFacade] = None
_ert: Optional[EnKFMain] = None
_config: Optional[ErtConfig] = None
_storage: Optional[StorageReader] = None

DEFAULT_SECURITY = Depends(security)


def init_facade() -> LibresFacade:
    # pylint: disable=global-statement
    global _libres_facade
    global _ert
    global _config

    configfile = os.environ["ERT_STORAGE_RES_CONFIG"]

    _config = ErtConfig.from_file(configfile)
    os.chdir(_config.config_path)
    _ert = EnKFMain(_config, read_only=True)
    _libres_facade = LibresFacade(_ert)
    return _libres_facade


def get_res(*, _: None = DEFAULT_SECURITY) -> LibresFacade:
    if _libres_facade is None:
        return init_facade()
    return _libres_facade


DEFAULT_LIBRESFACADE = Depends(get_res)


def get_storage(*, res: LibresFacade = DEFAULT_LIBRESFACADE) -> StorageReader:
    # pylint: disable=global-statement
    global _storage
    if _storage is None:
        return (_storage := open_storage(res.enspath))
    _storage.refresh()
    return _storage


def reset_res(*, _: None = DEFAULT_SECURITY) -> None:
    # pylint: disable=global-statement
    global _libres_facade
    if _libres_facade is not None:
        _libres_facade = None
    return _libres_facade
