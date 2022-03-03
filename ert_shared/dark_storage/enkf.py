import uuid
import os
from typing import Optional

import contextlib
from fastapi import Depends
from ert_storage.security import security

from ert_gui.ertnotifier import ErtNotifier
from ert_shared.ert_adapter import ErtAdapter
from ert_shared.libres_facade import LibresFacade
from res.enkf import EnKFMain, ResConfig

__all__ = ["LibresFacade", "get_res"]

ids = {}


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
    configfile = os.environ["ERT_STORAGE_RES_CONFIG"]
    config = ResConfig(configfile)
    os.chdir(config.config_path)
    ert = EnKFMain(config, strict=True)
    notifier = ErtNotifier(ert, config)
    adapter = ErtAdapter()
    with adapter.adapt(notifier):
        yield adapter.enkf_facade


def get_size(res, ensemble_name):
    return res.get_ensemble_size()


def get_active_realizations(ensemble_name, res):
    return res.get_active_realizations(ensemble_name)
