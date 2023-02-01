from cwrap import BaseCEnum


class EnkfTruncationType(BaseCEnum):
    TYPE_NAME = "enkf_truncation_type_enum"
    TRUNCATE_NONE = 0
    TRUNCATE_MIN = 1
    TRUNCATE_MAX = 2


EnkfTruncationType.addEnum("TRUNCATE_NONE", 0)
EnkfTruncationType.addEnum("TRUNCATE_MIN", 1)
EnkfTruncationType.addEnum("TRUNCATE_MAX", 2)
