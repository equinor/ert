from abc import ABC, abstractmethod


class Report(ABC):
    @abstractmethod
    def report(self, msg):
        """Method wil report messages given, when executed by job runner."""
        pass
