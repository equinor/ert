from __future__ import annotations

from typing import TYPE_CHECKING

from ert.config.parsing.queue_system import QueueSystem
from ert.scheduler.driver import Driver
from ert.scheduler.local_driver import LocalDriver
from ert.scheduler.scheduler import Scheduler

if TYPE_CHECKING:
    from ert.config.queue_config import QueueConfig


def create_driver(config: QueueConfig) -> Driver:
    if config.queue_system != QueueSystem.LOCAL:
        raise NotImplementedError("Only LOCAL driver is implemented")

    return LocalDriver()


__all__ = [
    "Scheduler",
    "Driver",
    "create_driver",
]
