from cwrap import BaseCEnum


class QueueDriverEnum(BaseCEnum):  # type: ignore
    TYPE_NAME = "queue_driver_enum"
    NULL_DRIVER = None
    LSF_DRIVER = None
    LOCAL_DRIVER = None
    TORQUE_DRIVER = None
    SLURM_DRIVER = None


QueueDriverEnum.addEnum("NULL_DRIVER", 0)
QueueDriverEnum.addEnum("LSF_DRIVER", 1)
QueueDriverEnum.addEnum("LOCAL_DRIVER", 2)
QueueDriverEnum.addEnum("TORQUE_DRIVER", 4)
QueueDriverEnum.addEnum("SLURM_DRIVER", 5)
