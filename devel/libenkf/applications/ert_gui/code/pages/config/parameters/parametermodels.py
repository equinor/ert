from enums import enkf_impl_type

class Model:

    def __init__(self, name):
        self.name = name
        self.data = {}

    def set(self, attr, value):
        self[attr] = value

    def __setitem__(self, attr, value):
        self.data[attr] = value

    def __getitem__(self, item):
        return self.data[item]


class FieldModel(Model):
    TYPE = enkf_impl_type.FIELD

    def __init__(self, name):
        Model.__init__(self, name)
        self.name = name

        self["type"] = "General"
        self["min"] = ""
        self["max"] = ""
        self["init"] = "None"
        self["output"] = "None"
        self["init_files"] = ""
        self["file_generated_by_enkf"] = ""
        self["file_loaded_by_enkf"] = ""

class KeywordModel(Model):
    TYPE = enkf_impl_type.GEN_KW

    def __init__(self, name):
        Model.__init__(self, name)
        self.name = name

        self["eclipse_file"] = ""
        self["template"] = ""
        self["priors"] = ""

class DataModel(Model):
    TYPE = enkf_impl_type.GEN_DATA

    #gen_data_file_format_type;
    #GEN_DATA_UNDEFINED = 0,
    ASCII = 1
    ASCII_TEMPLATE = 2
    BINARY_DOUBLE = 3
    BINARY_FLOAT = 4
    
    def __init__(self, name):
        Model.__init__(self, name)
        self.name = name

        self["input"] = "ASCII"
        self["output"] = "ASCII"
        self["eclipse_file"] = ""
        self["init_files"] = ""
        self["template"] = ""
        self["result_file"] = ""

class SummaryModel(Model):
    TYPE = enkf_impl_type.SUMMARY

    def __init__(self, name):
        Model.__init__(self, name)
        self.name = name