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

        self.min_std = PathChooser(self, "", "gen_kw_min_std", True)
        self.min_std.setter = lambda model, value: self.keywordModel.set("min_std", value)
        self.min_std.getter = lambda model: self.keywordModel["min_std"]

        self.template = PathChooser(self, "", "gen_kw_template", True, must_be_set=False)
        self.template.setter = lambda model, value: self.keywordModel.set("template", value)
        self.template.getter = lambda model: self.keywordModel["template"]

        self.enkf_outfile = PathChooser(self, "", "gen_kw_enkf_outfile", True, must_be_set=False)
        self.enkf_outfile.setter = lambda model, value: self.keywordModel.set("enkf_outfile", value)
        self.enkf_outfile.getter = lambda model: self.keywordModel["enkf_outfile"]

        self.init_file = PathChooser(self, "", "gen_kw_init_file", True, must_be_set=False)
        self.init_file.setter = lambda model, value: self.keywordModel.set("init_file", value)
        self.init_file.getter = lambda model: self.keywordModel["init_file"]

        layout.addRow("Min. std.:", self.min_std)
        layout.addRow("Template:", self.template)
        layout.addRow("EnKF outfile:", self.enkf_outfile)
        layout.addRow("Init file:", self.init_file)

        self.setLayout(layout)

    def setKeywordModel(self, keywordModel):
        self.keywordModel = keywordModel

        self.min_std.fetchContent()
        self.template.fetchContent()
        self.enkf_outfile.fetchContent()
        self.init_file.fetchContent()
