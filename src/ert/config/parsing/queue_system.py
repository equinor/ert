from enum import StrEnum, auto


def _ignore_case(enum, value):
    value = value.lower()
    for member in enum:
        if member == value:
            return member
    return None


class QueueSystem(StrEnum):
    LSF = auto()
    LOCAL = auto()
    TORQUE = auto()
    SLURM = auto()

    @classmethod
    def _missing_(cls, value):
        return _ignore_case(cls, value)


class QueueSystemWithGeneric(StrEnum):
    LSF = auto()
    LOCAL = auto()
    TORQUE = auto()
    SLURM = auto()
    GENERIC = auto()

    @classmethod
    def _missing_(cls, value):
        return _ignore_case(cls, value)
