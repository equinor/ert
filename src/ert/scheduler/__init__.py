from __future__ import annotations

from os import getuid
from pwd import getpwuid
from typing import TYPE_CHECKING

from ert.config.parsing.queue_system import QueueSystem

from .driver import Driver
from .job import JobState
from .local_driver import LocalDriver
from .lsf_driver import LsfDriver
from .openpbs_driver import OpenPBSDriver
from .scheduler import Scheduler
from .slurm_driver import SlurmDriver

if TYPE_CHECKING:
    from ert.config.queue_config import QueueConfig


def create_driver(config: QueueConfig) -> Driver:
    if config.queue_system == QueueSystem.LOCAL:
        return LocalDriver(**config.queue_options.driver_options)
    elif config.queue_system == QueueSystem.TORQUE:
        return OpenPBSDriver(**config.queue_options.driver_options)
    elif config.queue_system == QueueSystem.LSF:
        return LsfDriver(**config.queue_options.driver_options)
    elif config.queue_system == QueueSystem.SLURM:
        return SlurmDriver(
            **dict(
                {"user": getpwuid(getuid()).pw_name},
                **config.queue_options.driver_options,
            )
        )
    else:
        raise NotImplementedError(
            "Only LOCAL, SLURM, TORQUE and LSF drivers are implemented"
        )


__all__ = [
    "Driver",
    "JobState",
    "Scheduler",
    "create_driver",
]
