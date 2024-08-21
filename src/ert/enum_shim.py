import sys

if sys.version_info < (3, 11):
    from enum import Enum
    from enum import EnumMeta as EnumType

    class StrEnum(str, Enum):
        pass

else:
    from enum import EnumType, StrEnum

__all__ = ["EnumType", "StrEnum"]
