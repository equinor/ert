from enum import StrEnum, auto


def _ignore_case(enum: type[StrEnum], value: str) -> StrEnum | None:
    value = value.lower()
    for member in enum:
        if member.value.lower() == value:
            return member
    return None


class QueueSystem(StrEnum):
    LSF = auto()
    LOCAL = auto()
    TORQUE = auto()
    SLURM = auto()

    @classmethod
    def _missing_(cls, value: object) -> StrEnum | None:
        assert isinstance(value, str)
        return _ignore_case(cls, value)


class QueueSystemWithGeneric(StrEnum):
    LSF = auto()
    LOCAL = auto()
    TORQUE = auto()
    SLURM = auto()
    GENERIC = auto()

    @staticmethod
    def ert_config_case() -> str:
        return "upper"

    @classmethod
    def _missing_(cls, value: object) -> StrEnum | None:
        assert isinstance(value, str)
        return _ignore_case(cls, value)
