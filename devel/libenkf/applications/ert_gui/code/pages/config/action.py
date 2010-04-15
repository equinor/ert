# ----------------------------------------------------------------------------------------------
# Action tab
# ----------------------------------------------------------------------------------------------
from widgets.stringbox import StringBox

def createActionPage(configPanel, parent):
    configPanel.startPage("Action")

    r = configPanel.addRow(StringBox(parent, "Select case", "select_case"))
    r.getter = lambda ert : ert.getAttribute("select_case")
    r.setter = lambda ert, value : ert.setAttribute("select_case", value)

    configPanel.endPage()
