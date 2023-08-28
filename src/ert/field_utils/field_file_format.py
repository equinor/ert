import sys

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum


class FieldFileFormat(StrEnum):
    ROFF_BINARY = "roff_binary"
    ROFF_ASCII = "roff_ascii"
    ROFF = "roff"
    GRDECL = "grdecl"
    BGRDECL = "bgrdecl"


ROFF_FORMATS = (
    FieldFileFormat.ROFF_BINARY,
    FieldFileFormat.ROFF_ASCII,
    FieldFileFormat.ROFF,
)
