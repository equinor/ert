from abc import abstractmethod

from pydantic import BaseModel


class LoginProvider(BaseModel):
    """Base class for login provider plugins.

    Only covers the interactive login run by ``ert login``.
    """

    @abstractmethod
    def login(self) -> None:
        """Log in interactively, for example through a browser."""
