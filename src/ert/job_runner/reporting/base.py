from abc import ABC, abstractmethod
from ert.job_runner.reporting.message import Message


class Reporter(ABC):
    @abstractmethod
    def report(self, msg: Message):
        """Report a message."""
        pass
