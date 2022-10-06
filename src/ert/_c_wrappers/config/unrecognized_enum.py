from cwrap import BaseCEnum


class UnrecognizedEnum(BaseCEnum):
    TYPE_NAME = "config_unrecognized_enum"
    CONFIG_UNRECOGNIZED_IGNORE = None
    CONFIG_UNRECOGNIZED_WARN = None
    CONFIG_UNRECOGNIZED_ERROR = None
    CONFIG_UNRECOGNZIED_ADD = None


UnrecognizedEnum.addEnum("CONFIG_UNRECOGNIZED_IGNORE", 0)
UnrecognizedEnum.addEnum("CONFIG_UNRECOGNIZED_WARN", 1)
UnrecognizedEnum.addEnum("CONFIG_UNRECOGNIZED_ERROR", 2)
UnrecognizedEnum.addEnum("CONFIG_UNRECOGNIZED_ADD", 3)
