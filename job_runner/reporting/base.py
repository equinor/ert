from abc import ABC, abstractmethod


class Reporter(ABC):
    @abstractmethod
    def report(self, msg):
        """Report a message."""
        pass
