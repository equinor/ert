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

        self.min_std = PathChooser(self, "", "gen_kw_min_std", True , must_be_set = False)
        self.modelWrap(self.min_std, "min_std")
        
        self.template = PathChooser(self, "", "gen_kw_template", True)
        self.modelWrap(self.template, "template")

        self.enkf_outfile = PathChooser(self, "", "gen_kw_enkf_outfile", True, must_be_set=False)
        self.modelWrap(self.enkf_outfile, "enkf_outfile")

        self.init_file = PathChooser(self, "", "gen_kw_init_file", True, must_be_set=False)
        self.modelWrap(self.init_file, "init_file")

        
        layout.addRow("Template:"     , self.template)
        layout.addRow("Include file:" , self.enkf_outfile)
        layout.addRow("Minimum std:"  , self.min_std)
        layout.addRow("Init files:"   , self.init_file)

        self.setLayout(layout)

    def setKeywordModel(self, keywordModel):
        self.keywordModel = keywordModel

        self.min_std.fetchContent()
        self.template.fetchContent()
        self.enkf_outfile.fetchContent()
        self.init_file.fetchContent()

    def modelWrap(self, widget, attribute):
        widget.setter = lambda model, value: self.keywordModel.set(attribute, value)
        widget.getter = lambda model: self.keywordModel[attribute]

