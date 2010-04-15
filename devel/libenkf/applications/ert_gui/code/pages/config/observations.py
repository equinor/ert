# ----------------------------------------------------------------------------------------------
# Observations tab
# ----------------------------------------------------------------------------------------------
from widgets.combochoice import ComboChoice
from widgets.pathchooser import PathChooser

def createObservationsPage(configPanel, parent):
    configPanel.startPage("Observations")

    r = configPanel.addRow(ComboChoice(parent, ["REFCASE_SIMULATED", "REFCASE_HISTORY"], "History source", "history_source"))
    r.getter = lambda ert : ert.getAttribute("history_source")
    r.setter = lambda ert, value : ert.setAttribute("history_source", value)

    r = configPanel.addRow(PathChooser(parent, "Observations config", "obs_config", True))
    r.getter = lambda ert : ert.getAttribute("obs_config")
    r.setter = lambda ert, value : ert.setAttribute("obs_config", value)

    configPanel.endPage()
