from cwrap import BaseCEnum


class QueueSystem(BaseCEnum):  # type: ignore
    TYPE_NAME = "queue_driver_enum"
    NULL = None
    LSF = None
    LOCAL = None
    TORQUE = None
    SLURM = None


QueueSystem.addEnum("NULL", 0)
QueueSystem.addEnum("LSF", 1)
QueueSystem.addEnum("LOCAL", 2)
QueueSystem.addEnum("TORQUE", 4)
QueueSystem.addEnum("SLURM", 5)
