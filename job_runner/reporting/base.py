from abc import ABC, abstractmethod
from job_runner.reporting.message import Message


class Reporter(ABC):
    @abstractmethod
    def report(self, msg: Message):
        """Report a message."""
        pass
