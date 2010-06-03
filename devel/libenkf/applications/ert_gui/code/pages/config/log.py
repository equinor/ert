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
    r.getter = lambda ert : ert.getAttribute("update_log_path")
    r.setter = lambda ert, value : ert.setAttribute("update_log_path", value)

    configPanel.endPage()
