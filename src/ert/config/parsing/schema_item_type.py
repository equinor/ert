from enum import StrEnum


class SchemaItemType(StrEnum):
    STRING = "STRING"
    INT = "INT"
    POSITIVE_INT = "POSITIVE_INT"
    FLOAT = "FLOAT"
    POSITIVE_FLOAT = "POSITIVE_FLOAT"
    PATH = "PATH"
    EXISTING_PATH = "EXISTING_PATH"
    # EXISTING_PATH_INLINE is a directive to the
    # schema validation to inline the contents of
    # the file.
    EXISTING_PATH_INLINE = "EXISTING_PATH_INLINE"
    BOOL = "BOOL"
    CONFIG = "CONFIG"
    BYTESIZE = "BYTESIZE"
    EXECUTABLE = "EXECUTABLE"
    ISODATE = "ISODATE"
    INVALID = "INVALID"
    RUNTIME_INT = "RUNTIME_INT"
    RUNTIME_FILE = "RUNTIME_FILE"
