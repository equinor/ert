from PyQt4 import QtGui, QtCore
from widgets.combochoice import ComboChoice
from widgets.stringbox import DoubleBox
from widgets.pathchooser import PathChooser


class DataModel:
    name = ""
    input = "ASCII"
    output = "ASCII"
    eclipse_file = ""
    init_files = ""
    template = ""
    result_file = ""

    def __init__(self, name):
        self.name = name


class DataPanel(QtGui.QFrame):
    dataModel = DataModel("")

    def __init__(self, parent):
        QtGui.QFrame.__init__(self, parent)

        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.setFrameShadow(QtGui.QFrame.Plain)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)


        self.input = ComboChoice(self, ["ASCII", "BINARY_FLOAT", "BINARY_DOUBLE"], "", "param_init")
        self.input.setter = lambda model, value: setattr(self.dataModel, "input", value)
        self.input.getter = lambda model: self.dataModel.input

        self.output = ComboChoice(self, ["ASCII", "ASCII_TEMPLATE", "BINARY_FLOAT", "BINARY_DOUBLE"], "", "param_output")
        self.output.setter = lambda model, value: setattr(self.dataModel, "output", value)
        self.output.getter = lambda model: self.dataModel.output

        self.eclipse_file = PathChooser(self, "", "gen_data_eclipse_file", True)
        self.eclipse_file.setter = lambda model, value: setattr(self.dataModel, "eclipse_file", value)
        self.eclipse_file.getter = lambda model: self.dataModel.eclipse_file

        self.init_files = PathChooser(self, "", "gen_data_init_files", True)
        self.init_files.setter = lambda model, value: setattr(self.dataModel, "init_files", value)
        self.init_files.getter = lambda model: self.dataModel.init_files

        self.template = PathChooser(self, "", "gen_data_template", True)
        self.template.setter = lambda model, value: setattr(self.dataModel, "template", value)
        self.template.getter = lambda model: self.dataModel.template

        self.result_file = PathChooser(self, "", "gen_data_result_file", True)
        self.result_file.setter = lambda model, value: setattr(self.dataModel, "result_file", value)
        self.result_file.getter = lambda model: self.dataModel.result_file



        layout.addRow("Input:", self.input)
        layout.addRow("Output:", self.output)
        layout.addRow("Eclipse file:", self.eclipse_file)
        layout.addRow("Init files:", self.init_files)
        layout.addRow("Template:", self.template)
        layout.addRow("Result File:", self.result_file)

        self.setLayout(layout)

    def setDataModel(self, dataModel):
        self.dataModel = dataModel

        self.input.fetchContent()
        self.output.fetchContent()
        self.eclipse_file.fetchContent()
        self.init_files.fetchContent()
        self.template.fetchContent()
        self.result_file.fetchContent()
