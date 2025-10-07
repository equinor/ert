from __future__ import annotations

from os import getuid
from pwd import getpwuid
from typing import TYPE_CHECKING, assert_never

from ert.config import QueueSystem

from .driver import Driver
from .job import JobState
from .local_driver import LocalDriver
from .lsf_driver import LsfDriver
from .openpbs_driver import OpenPBSDriver
from .scheduler import Scheduler
from .slurm_driver import SlurmDriver

if TYPE_CHECKING:
    from ert.config.queue_config import QueueOptions


def create_driver(queue_options: QueueOptions) -> Driver:
    match queue_options.name:
        case QueueSystem.LOCAL:
            return LocalDriver()
        case QueueSystem.TORQUE:
            return OpenPBSDriver(**queue_options.driver_options)
        case QueueSystem.LSF:
            return LsfDriver(**queue_options.driver_options)
        case QueueSystem.SLURM:
            return SlurmDriver(
                **dict(
                    {"user": getpwuid(getuid()).pw_name},
                    **queue_options.driver_options,
                )
            )
        case default:
            assert_never(default)


__all__ = [
    "Driver",
    "JobState",
    "Scheduler",
    "create_driver",
]
