# ----------------------------------------------------------------------------------------------
# Log tab
# ----------------------------------------------------------------------------------------------
from widgets.pathchooser import PathChooser
from widgets.spinnerwidgets import IntegerSpinner
import ertwrapper

def createLogPage(configPanel, widget):
    configPanel.startPage("Log")

    r = configPanel.addRow(PathChooser(widget, "Log file", "log_file", True))
    r.initialize = lambda ert : [ert.prototype("char* log_get_filename(long)", lib=ert.util),
                                 ert.prototype("void log_reset_filename(long, char*)", lib=ert.util)]
    r.getter = lambda ert : ert.util.log_get_filename(ert.logh)
    r.setter = lambda ert, value : ert.util.log_reset_filename(ert.logh, value)

    r = configPanel.addRow(IntegerSpinner(widget, "Log level", "log_level", 0, 1000))
    r.initialize = lambda ert : [ert.prototype("int log_get_level(long)", lib=ert.util),
                                 ert.prototype("void log_set_level(long, int)", lib=ert.util)]
    r.getter = lambda ert : ert.util.log_get_level(ert.logh)
    r.setter = lambda ert, value : ert.util.log_set_level(ert.logh, value)

    r = configPanel.addRow(PathChooser(widget, "Update log path", "update_log_path"))
    r.initialize = lambda ert : [ert.prototype("long enkf_main_get_analysis_config(long)"),
                                 ert.prototype("char* analysis_config_get_log_path(long)"),
                                 ert.prototype("void analysis_config_set_log_path(long, char*)")]

    def get_update_log_path(ert):
        analysis_config = ert.enkf.enkf_main_get_analysis_config(ert.main)
        return ert.enkf.analysis_config_get_log_path(analysis_config)

    r.getter = get_update_log_path

    def set_update_log_path(ert, value):
        analysis_config = ert.enkf.enkf_main_get_analysis_config(ert.main)
        ert.enkf.analysis_config_set_log_path(analysis_config, str(value))

    r.setter = set_update_log_path

    configPanel.endPage()
