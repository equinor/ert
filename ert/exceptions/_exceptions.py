from typing import Optional


class ErtError(Exception):
    """Base class for exceptions in this module."""

    pass


class IllegalWorkspaceOperation(ErtError):
    def __init__(self, message: str) -> None:
        self.message = message


class IllegalWorkspaceState(ErtError):
    def __init__(self, message: str) -> None:
        self.message = message


class NonExistentExperiment(IllegalWorkspaceOperation):
    def __init__(self, message: str) -> None:
        self.message = message


class ConfigValidationError(ErtError):
    def __init__(self, message: str, source: Optional[str] = None) -> None:
        self.message = message
        self.source = source


class FileExistsException(ErtError):
    """Indicates an exceptional case where a file existed, and overwriting or
    appending to it could lead to data loss."""

    def __init__(self, message: str) -> None:
        self.message = message


class StorageError(ErtError):
    def __init__(self, message: str) -> None:
        self.message = message


class ElementExistsError(StorageError):
    def __init__(self, message: str) -> None:
        self.message = message


class ElementMissingError(StorageError):
    def __init__(self, message: str) -> None:
        self.message = message


class ExperimentError(ErtError):
    def __init__(self, message: str) -> None:
        self.message = message
