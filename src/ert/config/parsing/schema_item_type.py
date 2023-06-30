from enum import Enum
from typing import Dict

_old_to_new: Dict[int, "SchemaItemType"] = {}


class SchemaItemType(str, Enum):
    STRING = "STRING"
    INT = "INT"
    FLOAT = "FLOAT"
    PATH = "PATH"
    EXISTING_PATH = "EXISTING_PATH"
    BOOL = "BOOL"
    CONFIG = "CONFIG"
    BYTESIZE = "BYTESIZE"
    EXECUTABLE = "EXECUTABLE"
    ISODATE = "ISODATE"
    INVALID = "INVALID"
    RUNTIME_INT = "RUNTIME_INT"
    RUNTIME_FILE = "RUNTIME_FILE"

    @classmethod
    def from_content_type_enum(cls, old: int) -> "SchemaItemType":
        if old not in _old_to_new:
            raise ValueError("Invalid old enum value")

        return _old_to_new[old]


_old_to_new = {
    1: SchemaItemType.STRING,
    2: SchemaItemType.INT,
    4: SchemaItemType.FLOAT,
    8: SchemaItemType.PATH,
    16: SchemaItemType.EXISTING_PATH,
    32: SchemaItemType.BOOL,
    64: SchemaItemType.CONFIG,
    128: SchemaItemType.BYTESIZE,
    256: SchemaItemType.EXECUTABLE,
    512: SchemaItemType.ISODATE,
    1024: SchemaItemType.INVALID,
    2048: SchemaItemType.RUNTIME_INT,
    4096: SchemaItemType.RUNTIME_FILE,
}
