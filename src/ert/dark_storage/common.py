import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata

from fastapi import HTTPException

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


@contextmanager
def reraise_as_http_errors(
    custom_logger: logging.Logger = logger, details: dict[int, str] | None = None
) -> Iterator[None]:
    error_details = {404: "Ensemble not found", 500: "Internal server error"} | (
        details or {}
    )
    try:
        yield
    except KeyError as e:
        custom_logger.error(e)
        raise HTTPException(status_code=404, detail=error_details[404]) from e
    except ValueError as e:
        logger.error(e)
        raise HTTPException(status_code=404, detail="Data not found") from e
    except PermissionError as e:
        custom_logger.error(e)
        raise HTTPException(status_code=401, detail=str(e)) from e
    except Exception as ex:
        custom_logger.exception(ex)
        raise HTTPException(status_code=500, detail=error_details[500]) from ex
