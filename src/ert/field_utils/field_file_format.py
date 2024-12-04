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
