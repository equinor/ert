# ----------------------------------------------------------------------------------------------
# Observations tab
# ----------------------------------------------------------------------------------------------
from widgets.combochoice import ComboChoice
from widgets.pathchooser import PathChooser
from enums import history_source_type
from widgets.reloadbutton import ReloadButton

def createObservationsPage(configPanel, parent):
    configPanel.startPage("Observations")

    r = configPanel.addRow(ComboChoice(parent, history_source_type.values(), "History source", "config/observations/history_source"))
    r.initialize = lambda ert : [ert.prototype("int model_config_get_history_source(long)"),
                                 ert.prototype("void model_config_set_history_source(long, int)")]

    def get_history_source(ert):
        history_source = ert.enkf.model_config_get_history_source(ert.model_config)
        return history_source_type.resolveValue(history_source)

    r.getter = get_history_source

    def set_history_source(ert, value):
        history_source = history_source_type.resolveName(str(value))
        ert.enkf.model_config_get_history_source(ert.model_config, history_source.value())
        
    r.setter = set_history_source

    
    r = configPanel.addRow(PathChooser(parent, "Observations config", "config/observations/obs_config", True))
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


    r = configPanel.addRow(ReloadButton(parent, "Reload Observations", "config/observations/reload_observation", "Reload"))
    r.initialize = lambda ert : [ert.prototype("void enkf_main_reload_obs(long)")]
    r.getter = lambda ert : ert.enkf.enkf_main_reload_obs(ert.main)
    

    configPanel.endPage()
