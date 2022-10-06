from cwrap import BaseCEnum


class GenDataFileType(BaseCEnum):
    TYPE_NAME = "gen_data_file_format_type"
    GEN_DATA_UNDEFINED = None
    ASCII = None  # The file is ASCII file with a vector of numbers formatted with "%g"
    ASCII_TEMPLATE = None  # The data is inserted into a user defined template file.


GenDataFileType.addEnum("GEN_DATA_UNDEFINED", 0)
GenDataFileType.addEnum("ASCII", 1)
GenDataFileType.addEnum("ASCII_TEMPLATE", 2)
