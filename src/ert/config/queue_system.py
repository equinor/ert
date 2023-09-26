from cwrap import BaseCEnum


class QueueSystem(BaseCEnum):  # type: ignore
    TYPE_NAME = "queue_driver_enum"
    LSF = None
    LOCAL = None
    TORQUE = None
    SLURM = None


QueueSystem.addEnum("LSF", 1)
QueueSystem.addEnum("LOCAL", 2)
QueueSystem.addEnum("TORQUE", 4)
QueueSystem.addEnum("SLURM", 5)
