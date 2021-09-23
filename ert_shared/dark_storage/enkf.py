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

    if "ERT_DARK_STORAGE_CONFIG" in os.environ:
        configfile = os.environ["ERT_DARK_STORAGE_CONFIG"]
    else:
        configs = []
        for file in os.listdir(os.getcwd()):
            if file.endswith(".ert"):
                configs.append(file)
        if len(configs) == 1:
            configfile = file
        elif len(configs) > 1:
            raise ValueError(
                "Found several configs in current directory. Please use environment variable ERT_DARK_STORAGE_CONFIG to specify one config"
            )
        else:
            raise ValueError(
                "Found no configs in current directory. Please use environment variable ERT_DARK_STORAGE_CONFIG to specify a config"
            )

    _config = ResConfig(str(configfile))
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


def get_size():
    res = get_res()
    return res.get_ensemble_size()
