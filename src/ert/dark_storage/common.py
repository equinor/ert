import logging
import os

from ert.dark_storage.exceptions import InternalServerError
from ert.storage import Storage, open_storage

logger = logging.getLogger(__name__)


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        try:
            return (_storage := open_storage(os.environ["ERT_STORAGE_ENS_PATH"]))
        except RuntimeError as err:
            raise InternalServerError(f"{err!s}") from err
    _storage.refresh()
    return _storage
