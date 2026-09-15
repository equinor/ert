from typing import Any

from fastapi import status


class ErtStorageError(RuntimeError):
    """Base error class for all the rest of errors"""

    __status_code__ = status.HTTP_200_OK

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, kwargs)


class InternalServerError(ErtStorageError):
    __status_code__ = status.HTTP_500_INTERNAL_SERVER_ERROR
