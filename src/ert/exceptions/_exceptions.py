class ErtError(Exception):
    """Base class for exceptions in this module."""

    pass


class NonExistentExperiment(ErtError):
    def __init__(self, message: str) -> None:
        self.message = message


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
