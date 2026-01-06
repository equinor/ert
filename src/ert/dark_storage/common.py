import logging
import os
import re
from importlib import metadata

from ert.dark_storage.exceptions import InternalServerError
from ert.storage import (
    ErtStorageException,
    ErtStoragePermissionError,
    Storage,
    open_storage,
)

logger = logging.getLogger(__name__)


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        try:
            return (_storage := open_storage(os.environ["ERT_STORAGE_ENS_PATH"]))
        except ErtStoragePermissionError as err:
            logger.error(f"Permission error accessing storage: {err!s}")
            raise InternalServerError("Permission error accessing storage") from None
        except ErtStorageException as err:
            logger.exception(f"Error accessing storage: {err!s}")
            raise InternalServerError("Error accessing storage") from None
    _storage.reload()
    return _storage


def get_storage_api_version() -> str:
    major = minor = "0"
    match = re.match(r"(\d+)\.(\d+)", metadata.version("ert"))
    if match:
        major, minor = match.groups()
    return f"{major}.{minor}"
