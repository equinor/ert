from cwrap import BaseCEnum


class EnkfFieldFileFormatEnum(BaseCEnum):
    TYPE_NAME = "enkf_field_file_format_enum"
    UNDEFINED_FORMAT = None
    RMS_ROFF_FILE = None
    ECL_KW_FILE = None
    ECL_KW_FILE_ACTIVE_CELLS = None
    ECL_KW_FILE_ALL_CELLS = None
    ECL_GRDECL_FILE = None
    ECL_FILE = None
    FILE_FORMAT_NULL = None


EnkfFieldFileFormatEnum.addEnum("UNDEFINED_FORMAT", 0)
EnkfFieldFileFormatEnum.addEnum("RMS_ROFF_FILE", 1)
EnkfFieldFileFormatEnum.addEnum("ECL_KW_FILE", 2)
EnkfFieldFileFormatEnum.addEnum("ECL_KW_FILE_ACTIVE_CELLS", 3)
EnkfFieldFileFormatEnum.addEnum("ECL_KW_FILE_ALL_CELLS", 4)
EnkfFieldFileFormatEnum.addEnum("ECL_GRDECL_FILE", 5)
EnkfFieldFileFormatEnum.addEnum("ECL_FILE", 6)
EnkfFieldFileFormatEnum.addEnum("FILE_FORMAT_NULL", 7)
