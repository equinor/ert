from enum import Enum, auto


class SchemaType(Enum):
    CONFIG_STRING = auto()
    CONFIG_INT = auto()
    CONFIG_FLOAT = auto()
    CONFIG_PATH = auto()
    CONFIG_EXISTING_PATH = auto()
    CONFIG_BOOL = auto()
    CONFIG_CONFIG = auto()
    CONFIG_BYTESIZE = auto()
    CONFIG_EXECUTABLE = auto()
    CONFIG_ISODATE = auto()
    CONFIG_INVALID = auto()
    CONFIG_RUNTIME_INT = auto()
    CONFIG_RUNTIME_FILE = auto()
