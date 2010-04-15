# ----------------------------------------------------------------------------------------------
# dbase tab
# ----------------------------------------------------------------------------------------------
from widgets.combochoice import ComboChoice
from widgets.pathchooser import PathChooser

def createDbasePage(configPanel, parent):
    configPanel.startPage("dbase")

    r = configPanel.addRow(ComboChoice(parent, ["BLOCK_FS", "PLAIN"], "dbase type", "dbase_type"))
    r.getter = lambda ert : ert.getAttribute("dbase_type")
    r.setter = lambda ert, value : ert.setAttribute("dbase_type", value)

    r = configPanel.addRow(PathChooser(parent, "enspath", "enspath"))
    r.getter = lambda ert : ert.getAttribute("enspath")
    r.setter = lambda ert, value : ert.setAttribute("enspath", value)

    configPanel.endPage()