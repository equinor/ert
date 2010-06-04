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
    r.initialize = lambda ert : [ert.prototype("long enkf_main_get_obs(long)"),
                                 ert.prototype("char* enkf_obs_get_config_file(long)"),
                                 ert.prototype("void enkf_main_load_obs(long, char*)")]

    def get_obs(ert):
        obs = ert.enkf.enkf_main_get_obs(ert.main)
        return ert.enkf.enkf_obs_get_config_file(obs)

    r.getter = get_obs


    def set_obs(ert, value):
        ert.enkf.enkf_main_load_obs(ert.main, str(value))
    r.setter = set_obs

    configPanel.endPage()
