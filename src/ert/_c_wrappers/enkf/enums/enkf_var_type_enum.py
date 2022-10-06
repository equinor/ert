from cwrap import BaseCEnum


class EnkfVarType(BaseCEnum):
    TYPE_NAME = "enkf_var_type_enum"
    INVALID_VAR = None
    PARAMETER = None
    DYNAMIC_RESULT = None
    STATIC_STATE = None
    INDEX_STATE = None
    EXT_PARAMETER = None


EnkfVarType.addEnum("INVALID_VAR", 0)
EnkfVarType.addEnum("PARAMETER", 1)
EnkfVarType.addEnum("DYNAMIC_RESULT", 4)
EnkfVarType.addEnum("STATIC_STATE", 8)
EnkfVarType.addEnum("INDEX_STATE", 16)
EnkfVarType.addEnum("EXT_PARAMETER", 32)
