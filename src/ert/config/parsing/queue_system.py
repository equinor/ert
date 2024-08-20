from ert.enum_shim import StrEnum


class QueueSystem(StrEnum):
    LSF = "LSF"
    LOCAL = "LOCAL"
    TORQUE = "TORQUE"
    SLURM = "SLURM"


class QueueSystemWithGeneric(StrEnum):
    LSF = "LSF"
    LOCAL = "LOCAL"
    TORQUE = "TORQUE"
    SLURM = "SLURM"
    GENERIC = "GENERIC"
