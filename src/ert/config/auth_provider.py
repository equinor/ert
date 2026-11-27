from abc import abstractmethod

from pydantic import BaseModel


class AuthProvider(BaseModel):
    """Base class for auth provider plugins.

    Covers the interactive login and logout run by ``ert auth login`` and
    ``ert auth logout``.
    """

    @abstractmethod
    def login(self) -> None:
        """Log in interactively, for example through a browser."""

    @abstractmethod
    def logout(self) -> None:
        """Log out, for example by clearing cached tokens."""
