from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ert.config.parsing.queue_system import QueueSystem
from ert.scheduler.driver import Driver
from ert.scheduler.local_driver import LocalDriver
from ert.scheduler.scheduler import Scheduler
from ert.scheduler.torque_driver import TorqueDriver

if TYPE_CHECKING:
    from ert.config.queue_config import QueueConfig


def create_driver(config: QueueConfig) -> Driver:
    if config.queue_system == QueueSystem.LOCAL:
        return LocalDriver()
    elif config.queue_system == QueueSystem.TORQUE:
        queue_name: Optional[str] = None
        for key, val in config.queue_options.get(QueueSystem.TORQUE, []):
            if key == "QUEUE":
                queue_name = val

        return TorqueDriver(queue_name=queue_name)
    else:
        raise NotImplementedError("Only LOCAL and TORQUE drivers are implemented")


__all__ = [
    "Scheduler",
    "Driver",
    "create_driver",
]
