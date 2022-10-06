from cwrap import BaseCEnum


class FieldTypeEnum(BaseCEnum):
    TYPE_NAME = "field_type_enum"
    ECLIPSE_RESTART = None
    ECLIPSE_PARAMETER = None
    GENERAL = None
    UNKNOWN_FIELD_TYPE = None


FieldTypeEnum.addEnum("ECLIPSE_RESTART", 1)
FieldTypeEnum.addEnum("ECLIPSE_PARAMETER", 2)
FieldTypeEnum.addEnum("GENERAL", 3)
FieldTypeEnum.addEnum("UNKNOWN_FIELD_TYPE", 4)
