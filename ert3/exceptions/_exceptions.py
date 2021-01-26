class ErtError(Exception):
    """Base class for exceptions in this module."""

    pass


class IllegalWorkspaceOperation(ErtError):
    def __init__(self, message):
        self.message = "{}".format(message)
