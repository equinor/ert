from cwrap import BaseCEnum

from ert.enum_shim import StrEnum


class QueueDriverEnum(BaseCEnum):  # type: ignore
    TYPE_NAME = "queue_driver_enum"
    LSF = None
    LOCAL = None
    TORQUE = None
    SLURM = None
    GENERIC = None


QueueDriverEnum.addEnum("LSF", 1)
QueueDriverEnum.addEnum("LOCAL", 2)
QueueDriverEnum.addEnum("TORQUE", 4)
QueueDriverEnum.addEnum("SLURM", 5)
QueueDriverEnum.addEnum("GENERIC", 6)


class QueueSystem(StrEnum):
    LSF = "LSF"
    LOCAL = "LOCAL"
    TORQUE = "TORQUE"
    SLURM = "SLURM"
    GENERIC = "GENERIC"

    def to_c_enum(self) -> QueueDriverEnum:
        return QueueDriverEnum.from_string(self.name)
