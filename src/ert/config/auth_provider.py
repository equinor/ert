from abc import abstractmethod

from pydantic import BaseModel


class AuthProvider(BaseModel):
    """Base class for auth provider plugins.

    Only covers the interactive login run by ``ert login``.
    """

    @abstractmethod
    def login(self) -> None:
        """Log in interactively, for example through a browser."""
