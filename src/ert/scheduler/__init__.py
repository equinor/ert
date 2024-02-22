from __future__ import annotations

from typing import TYPE_CHECKING

from ert.config.parsing.queue_system import QueueSystem
from ert.scheduler.driver import Driver
from ert.scheduler.local_driver import LocalDriver
from ert.scheduler.lsf_driver import LsfDriver
from ert.scheduler.openpbs_driver import OpenPBSDriver
from ert.scheduler.scheduler import Scheduler

if TYPE_CHECKING:
    from ert.config.queue_config import QueueConfig


def create_driver(config: QueueConfig) -> Driver:
    if config.queue_system == QueueSystem.LOCAL:
        return LocalDriver()
    elif config.queue_system == QueueSystem.TORQUE:
        queue_config = {
            key: value
            for key, value in config.queue_options.get(QueueSystem.TORQUE, [])
        }
        return OpenPBSDriver(
            queue_name=queue_config.get("QUEUE"),
            memory_per_job=queue_config.get("MEMORY_PER_JOB"),
            num_nodes=int(queue_config.get("NUM_NODES", 1)),
            num_cpus_per_node=int(queue_config.get("NUM_CPUS_PER_NODE", 1)),
            job_prefix=queue_config.get("JOB_PREFIX"),
        )
    elif config.queue_system == QueueSystem.LSF:
        queue_config = {
            key: value for key, value in config.queue_options.get(QueueSystem.LSF, [])
        }
        return LsfDriver(
            bsub_cmd=queue_config.get("BSUB_CMD"),
            bkill_cmd=queue_config.get("BJOBS_CMD"),
            bjobs_cmd=queue_config.get("BJOBS_CMD"),
        )
    else:
        raise NotImplementedError("Only LOCAL, TORQUE and LSF drivers are implemented")


__all__ = [
    "Scheduler",
    "Driver",
    "create_driver",
]
