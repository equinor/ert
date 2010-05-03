from PyQt4 import QtGui, QtCore
from widgets.combochoice import ComboChoice
from widgets.stringbox import DoubleBox
from widgets.pathchooser import PathChooser
from pages.config.parameters.parametermodels import KeywordModel

class KeywordPanel(QtGui.QFrame):
    def __init__(self, parent):
        QtGui.QFrame.__init__(self, parent)

        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.setFrameShadow(QtGui.QFrame.Plain)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        self.keywordModel = KeywordModel("")

        self.eclipse_file = PathChooser(self, "", "gen_kw_eclipse_file", True)
        self.eclipse_file.setter = lambda model, value: self.keywordModel.set("eclipse_file", value)
        self.eclipse_file.getter = lambda model: self.keywordModel["eclipse_file"]

        self.template = PathChooser(self, "", "gen_kw_template", True, must_be_set=False)
        self.template.setter = lambda model, value: self.keywordModel.set("template", value)
        self.template.getter = lambda model: self.keywordModel["template"]

        self.priors = PathChooser(self, "", "gen_kw_result_file", True, must_be_set=False)
        self.priors.setter = lambda model, value: self.keywordModel.set("priors", value)
        self.priors.getter = lambda model: self.keywordModel["priors"]

        layout.addRow("Eclipse file:", self.eclipse_file)
        layout.addRow("Template:", self.template)
        layout.addRow("Priors:", self.priors)

        self.setLayout(layout)

    def setKeywordModel(self, keywordModel):
        self.keywordModel = keywordModel

        self.eclipse_file.fetchContent()
        self.template.fetchContent()
        self.priors.fetchContent()
