from typing import Any

from fastapi import status


class ErtStorageError(RuntimeError):
    """
    Base error class for all the rest of errors
    """

    __status_code__ = status.HTTP_200_OK

    def __init__(self, message: str, **kwargs: Any):
        super().__init__(message, kwargs)


class NotFoundError(ErtStorageError):
    __status_code__ = status.HTTP_404_NOT_FOUND


class ConflictError(ErtStorageError):
    __status_code__ = status.HTTP_409_CONFLICT


class ExpectationError(ErtStorageError):
    __status_code__ = status.HTTP_417_EXPECTATION_FAILED


class UnprocessableError(ErtStorageError):
    __status_code__ = status.HTTP_422_UNPROCESSABLE_ENTITY


class InternalServerError(ErtStorageError):
    __status_code__ = status.HTTP_500_INTERNAL_SERVER_ERROR
