#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'observations.py' is part of ERT - Ensemble based Reservoir Tool. 
#   
#  ERT is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published by 
#  the Free Software Foundation, either version 3 of the License, or 
#  (at your option) any later version. 
#   
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or 
#  FITNESS FOR A PARTICULAR PURPOSE.   
#   
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
#  for more details. 


# ----------------------------------------------------------------------------------------------
# Observations tab
# ----------------------------------------------------------------------------------------------
from ert_gui.widgets.combochoice import ComboChoice
from ert_gui.widgets.pathchooser import PathChooser
from ert.ert.enums import history_source_type
from ert_gui.widgets.reloadbutton import ReloadButton

def createObservationsPage(configPanel, parent):
    configPanel.startPage("Observations")

    r = configPanel.addRow(ComboChoice(parent, history_source_type.values(), "History source", "config/observations/history_source"))

    def get_history_source(ert):
        history_source = ert.main.model_config.get_history_source
        return history_source_type.resolveValue(history_source)

    r.getter = get_history_source

    def set_history_source(ert, value):
        history_source = history_source_type.resolveName(str(value))
        ert.main.model_config.set_history_source( history_source.value())
        
    r.setter = set_history_source

    
    r = configPanel.addRow(PathChooser(parent, "Observations config", "config/observations/obs_config", True))

    def get_obs(ert):
        obs = ert.main.get_obs
        return obs.get_config_file

    r.getter = get_obs


    def set_obs(ert, value):
        ert.main.load_obs( str(value))
    r.setter = set_obs


    r = configPanel.addRow(ReloadButton(parent, "Reload Observations", "config/observations/reload_observation", "Reload"))
    r.getter = lambda ert : ert.main.reload_obs
    

    configPanel.endPage()
