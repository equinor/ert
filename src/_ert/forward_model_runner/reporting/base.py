from abc import ABC, abstractmethod

from _ert.forward_model_runner.reporting.message import Message


class Reporter(ABC):
    @abstractmethod
    async def report(self, msg: Message):
        """Report a message."""

    @abstractmethod
    def cancel(self):
        """Safely shut down the reporter"""
