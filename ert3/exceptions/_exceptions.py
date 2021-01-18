import sys

class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class IllegalWorkspaceOperation(Error):
    """Exception raised when user tries to perform illegal workspace operation, for example, initialise it twice.
    """
    def __init__(self, message):
        self.args = "{}".format(message),
        sys.exit(self)
