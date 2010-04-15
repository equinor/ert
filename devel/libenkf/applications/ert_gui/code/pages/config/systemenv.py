
#----------------------------------------------------------------------------------------------
# System tab
# ----------------------------------------------------------------------------------------------
from widgets.pathchooser import PathChooser
from widgets.configpanel import ConfigPanel
from widgets.tablewidgets import KeywordTable
import ertwrapper

def createSystemPage(configPanel, parent):

   configPanel.startPage("System")

   r = configPanel.addRow(PathChooser(parent, "Job script", "job_script", True))
   r.initialize = lambda ert : [ert.setTypes("site_config_get_job_script", ertwrapper.c_char_p),
                                ert.setTypes("site_config_set_job_script", None, ertwrapper.c_char_p)]
   r.getter = lambda ert : ert.enkf.site_config_get_job_script(ert.site_config)
   r.setter = lambda ert, value : ert.enkf.site_config_set_job_script(ert.site_config, str(value))

   internalPanel = ConfigPanel(parent)
   internalPanel.startPage("setenv")

   r = internalPanel.addRow(KeywordTable(parent, "", "setenv"))
   r.initialize = lambda ert : [ert.setTypes("site_config_get_env_hash"),
                                ert.setTypes("site_config_clear_env", None),
                                ert.setTypes("site_config_setenv", None, [ertwrapper.c_char_p, ertwrapper.c_char_p])]
   r.getter = lambda ert : ert.getHash(ert.enkf.site_config_get_env_hash(ert.site_config))

   def setenv(ert, value):
       ert.enkf.site_config_clear_env(ert.site_config)
       for env in value:
           ert.enkf.site_config_setenv(ert.site_config, env[0], env[1])

   r.setter = setenv


   internalPanel.endPage()

   internalPanel.startPage("Update path")

   r = internalPanel.addRow(KeywordTable(parent, "", "update_path"))
   r.initialize = lambda ert : [ert.setTypes("site_config_get_path_variables"),
                                ert.setTypes("site_config_get_path_values"),
                                ert.setTypes("site_config_clear_pathvar", None),
                                ert.setTypes("site_config_update_pathvar", None, [ertwrapper.c_char_p, ertwrapper.c_char_p])]
   def get_update_path(ert):
       paths = ert.getStringList(ert.enkf.site_config_get_path_variables(ert.site_config))
       values =  ert.getStringList(ert.enkf.site_config_get_path_values(ert.site_config))

       return [[p, v] for p,v in zip(paths, values)]

   r.getter = get_update_path

   def update_pathvar(ert, value):
       ert.enkf.site_config_clear_pathvar(ert.site_config)

       for pathvar in value:
           ert.enkf.site_config_update_pathvar(ert.site_config, pathvar[0], pathvar[1])

   r.setter = update_pathvar

   internalPanel.endPage()

   internalPanel.startPage("Install job")

   r = internalPanel.addRow(KeywordTable(parent, "", "install_job"))
   r.getter = lambda ert : ert.getAttribute("install_job")
   r.setter = lambda ert, value : ert.setAttribute("install_job", value)

   internalPanel.endPage()
   configPanel.addRow(internalPanel)

   configPanel.endPage()
