import uuid
import os
from typing import Optional
from fastapi import Depends
from ert_storage.security import security
from ert_shared.libres_facade import LibresFacade
from res.enkf import EnKFMain, ResConfig

__all__ = ["LibresFacade", "get_res"]


_libres_facade: Optional[LibresFacade] = None
_ert: Optional[EnKFMain] = None
_config: Optional[ResConfig] = None
ids = {}


def init_facade() -> None:
    global _libres_facade
    global _ert
    global _config

    configfile = os.environ["ERT_STORAGE_RES_CONFIG"]

    _config = ResConfig(configfile)
    os.chdir(_config.config_path)
    _ert = EnKFMain(_config, strict=True)
    _libres_facade = LibresFacade(_ert)


def get_id(type, name):
    if type not in ids:
        ids[type] = {}
    if name not in ids[type]:
        ids[type][name] = uuid.uuid4()
    return ids[type][name]


def get_name(type, uuid):
    if type not in ids:
        ids[type] = {}
    for name, id in ids[type].items():
        if str(id) == str(uuid):
            return name
    raise ValueError(f"No such uuid for type {type}")


def get_res(*, _: None = Depends(security)) -> LibresFacade:
    global _libres_facade
    if _libres_facade is None:
        init_facade()
    return _libres_facade


def get_size(res: LibresFacade):
    return res.get_ensemble_size()


def get_active_realizations(res: LibresFacade, ensemble_name):
    return res.get_active_realizations(ensemble_name)
