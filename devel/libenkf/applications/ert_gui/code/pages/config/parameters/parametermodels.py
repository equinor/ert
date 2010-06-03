from enums import enkf_impl_type, field_type
from PyQt4.QtCore import QObject
from PyQt4.Qt import SIGNAL

class Model(QObject):

    def __init__(self, name):
        QObject.__init__(self)
        self.name = name
        self.data = {}
        self.valid = True

    def set(self, attr, value):
        self[attr] = value

    def __setitem__(self, attr, value):
        self.data[attr] = value
        self.emit(SIGNAL("modelChanged(Model)"), self)

    def __getitem__(self, item):
        return self.data[item]

    def isValid(self):
        return self.valid

    def setValid(self, valid):
        self.valid = valid

    def getName(self):
        return self.name


class FieldModel(Model):
    TYPE = enkf_impl_type.FIELD

    def __init__(self, name):
        Model.__init__(self, name)
        self.name = name

        self["type"] = field_type.GENERAL
        self["min"] = ""
        self["max"] = ""
        self["init"] = "None"
        self["output"] = "None"
        self["init_files"] = ""
        self["enkf_outfile"] = ""
        self["enkf_infile"] = ""
        self["min_std"] = ""

class KeywordModel(Model):
    TYPE = enkf_impl_type.GEN_KW

    def __init__(self, name):
        Model.__init__(self, name)
        self.name = name

        self["min_std"] = ""
        self["enkf_outfile"] = ""
        self["template"] = ""
        self["init_files"] = ""
        self["parameter_file"] = ""

class DataModel(Model):
    TYPE = enkf_impl_type.GEN_DATA
    
    def __init__(self, name):
        Model.__init__(self, name)
        self.name = name

        self["input_format"] = ""
        self["output_format"] = ""
        self["template_file"] = ""
        self["template_key"] = ""
        self["init_file_fmt"] = ""

class SummaryModel(Model):
    TYPE = enkf_impl_type.SUMMARY

    def __init__(self, name):
        Model.__init__(self, name)
        self.name = name