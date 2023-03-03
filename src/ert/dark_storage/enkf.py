import os
import uuid
from typing import Optional

from ert_storage.security import security
from fastapi import Depends

from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.libres_facade import LibresFacade

__all__ = ["LibresFacade", "get_res"]


_libres_facade: Optional[LibresFacade] = None
_ert: Optional[EnKFMain] = None
_config: Optional[ErtConfig] = None
ids = {}


def init_facade() -> None:
    # pylint: disable=global-statement
    global _libres_facade
    global _ert
    global _config

    configfile = os.environ["ERT_STORAGE_RES_CONFIG"]

    _config = ErtConfig.from_file(configfile)
    os.chdir(_config.config_path)
    _ert = EnKFMain(_config, read_only=True)
    _libres_facade = LibresFacade(_ert)


def get_id(type_key, name):
    if type_key not in ids:
        ids[type_key] = {}
    if name not in ids[type_key]:
        ids[type_key][name] = uuid.uuid4()
    return ids[type_key][name]


def get_name(type_key, type_uuid):
    if type_key not in ids:
        ids[type_key] = {}
    for name, name_id in ids[type_key].items():
        if str(name_id) == str(type_uuid):
            return name
    raise ValueError(f"No such uuid for type {type_key}")


def get_res(*, _: None = Depends(security)) -> LibresFacade:
    if _libres_facade is None:
        init_facade()
    return _libres_facade


def reset_res(*, _: None = Depends(security)) -> None:
    # pylint: disable=global-statement
    global _libres_facade
    if _libres_facade is not None:
        _libres_facade = None
    return _libres_facade


def get_size(res: LibresFacade):
    return res.get_ensemble_size()


def get_active_realizations(res: LibresFacade, ensemble_name):
    return res.get_active_realizations(ensemble_name)
