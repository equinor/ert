from cwrap import BaseCEnum


class LoadFailTypeEnum(BaseCEnum):
    TYPE_NAME = "load_fail_type"
    LOAD_FAIL_SILENT = None
    LOAD_FAIL_WARN = None
    LOAD_FAIL_EXIT = None


LoadFailTypeEnum.addEnum("LOAD_FAIL_SILENT", 0)
LoadFailTypeEnum.addEnum("LOAD_FAIL_WARN", 2)
LoadFailTypeEnum.addEnum("LOAD_FAIL_EXIT", 4)
