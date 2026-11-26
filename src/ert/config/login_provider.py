from abc import abstractmethod

from pydantic import BaseModel


class LoginProvider(BaseModel):
    """Base class for login provider plugins."""

    @abstractmethod
    def login(self) -> None:
        """Interactive login."""
