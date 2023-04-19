from cwrap import BaseCEnum


class FieldFileFormatType(BaseCEnum):
    TYPE_NAME = "field_file_format_type_enum"
    UNDEFINED_FORMAT = None
    RMS_ROFF_FILE = None
    ECL_KW_FILE = None
    ECL_KW_FILE_ALL_CELLS = None
    ECL_GRDECL_FILE = None
    FILE_FORMAT_NULL = None


FieldFileFormatType.addEnum("UNDEFINED_FORMAT", 0)
FieldFileFormatType.addEnum("RMS_ROFF_FILE", 1)
FieldFileFormatType.addEnum("ECL_KW_FILE", 2)
FieldFileFormatType.addEnum("ECL_KW_FILE_ALL_CELLS", 4)
FieldFileFormatType.addEnum("ECL_GRDECL_FILE", 5)
FieldFileFormatType.addEnum("FILE_FORMAT_NULL", 7)
