from PyQt4 import QtGui, QtCore
from widgets.combochoice import ComboChoice
from widgets.stringbox import DoubleBox
from widgets.pathchooser import PathChooser
from pages.config.parameters.parametermodels import DataModel
import enums

class DataPanel(QtGui.QFrame):

    def __init__(self, parent):
        QtGui.QFrame.__init__(self, parent)

        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.setFrameShadow(QtGui.QFrame.Plain)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        self.dataModel = DataModel("")

        self.input = ComboChoice(self, enums.gen_data_file_format.INPUT_TYPES, "", "param_init")
        self.modelWrap(self.input, "input")

        self.output = ComboChoice(self, enums.gen_data_file_format.OUTPUT_TYPES, "", "param_output")
        self.modelWrap(self.output, "output")

        self.eclipse_file = PathChooser(self, "", "gen_data_eclipse_file", True , True)
        self.modelWrap(self.eclipse_file, "eclipse_file")

        self.init_files = PathChooser(self, "", "gen_data_init_files", True , True)
        self.modelWrap(self.init_files, "init_files")

        self.template = PathChooser(self, "", "gen_data_template", True , False)
        self.modelWrap(self.template, "template")

        self.result_file = PathChooser(self, "", "gen_data_result_file", True , False)
        self.modelWrap(self.result_file, "result_file")


        layout.addRow("Input:", self.input)
        layout.addRow("Output:", self.output)
        layout.addRow("Eclipse file:", self.eclipse_file)
        layout.addRow("Init files:", self.init_files)
        layout.addRow("Template:", self.template)
        layout.addRow("Result File:", self.result_file)

        self.setLayout(layout)

    def modelWrap(self, widget, attribute):
        widget.setter = lambda model, value: self.dataModel.set(attribute, value)
        widget.getter = lambda model: self.dataModel[attribute]

    def setDataModel(self, dataModel):
        self.dataModel = dataModel

        self.input.fetchContent()
        self.output.fetchContent()
        self.eclipse_file.fetchContent()
        self.init_files.fetchContent()
        self.template.fetchContent()
        self.result_file.fetchContent()
