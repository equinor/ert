from abc import ABC, abstractmethod

from _ert_forward_model_runner.reporting.message import Message


class Reporter(ABC):
    @abstractmethod
    def report(self, msg: Message):
        """Report a message."""
