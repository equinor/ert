from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends

from ert.dark_storage.exceptions import InternalServerError
from ert.dark_storage.security import security
from ert.storage import Storage, open_storage

__all__ = ["get_storage"]


_storage: Optional[Storage] = None

DEFAULT_SECURITY = Depends(security)


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        try:
            return (_storage := open_storage(os.environ["ERT_STORAGE_ENS_PATH"]))
        except RuntimeError as err:
            raise InternalServerError(f"{err!s}") from err
    _storage.refresh()
    return _storage
