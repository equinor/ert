# ----------------------------------------------------------------------------------------------
# Queue System tab
# ----------------------------------------------------------------------------------------------
from widgets.configpanel import ConfigPanel
from widgets.combochoice import ComboChoice
import ertwrapper
from widgets.stringbox import StringBox
from widgets.pathchooser import PathChooser
from widgets.spinnerwidgets import IntegerSpinner
from widgets.tablewidgets import KeywordTable

def createQueueSystemPage(configPanel, parent):
    configPanel.startPage("Queue System")

    r = configPanel.addRow(ComboChoice(parent, ["LSF", "RSH", "LOCAL"], "Queue system", "queue_system"))
    r.initialize = lambda ert : [ert.setTypes("site_config_get_queue_name", ertwrapper.c_char_p),
                                 ert.setTypes("site_config_set_job_queue", None, [ertwrapper.c_char_p])]
    r.getter = lambda ert : ert.enkf.site_config_get_queue_name(ert.site_config)
    r.setter = lambda ert, value : ert.enkf.site_config_set_job_queue(ert.site_config, str(value))

    internalPanel = ConfigPanel(parent)

    internalPanel.startPage("LSF")

    r = internalPanel.addRow(ComboChoice(parent, ["NORMAL", "FAST_LOCAL", "SHORT"], "Mode", "lsf_queue"))
    r.initialize = lambda ert : [ert.setTypes("site_config_get_lsf_queue", ertwrapper.c_char_p),
                                 ert.setTypes("site_config_set_lsf_queue", None, [ertwrapper.c_char_p])]
    r.getter = lambda ert : ert.enkf.site_config_get_lsf_queue(ert.site_config)
    r.setter = lambda ert, value : ert.enkf.site_config_set_lsf_queue(ert.site_config, str(value))

    r = internalPanel.addRow(IntegerSpinner(parent, "Max running", "max_running_lsf", 1, 1000))
    r.initialize = lambda ert : [ert.setTypes("site_config_get_max_running_lsf", ertwrapper.c_int),
                                 ert.setTypes("site_config_set_max_running_lsf", None, [ertwrapper.c_int])]
    r.getter = lambda ert : ert.enkf.site_config_get_max_running_lsf(ert.site_config)
    r.setter = lambda ert, value : ert.enkf.site_config_set_max_running_lsf(ert.site_config, value)

    r = internalPanel.addRow(StringBox(parent, "Resources", "lsf_resources"))
    r.initialize = lambda ert : [ert.setTypes("site_config_get_lsf_request", ertwrapper.c_char_p),
                                 ert.setTypes("site_config_set_lsf_request", None, [ertwrapper.c_char_p])]
    r.getter = lambda ert : ert.enkf.site_config_get_lsf_request(ert.site_config)
    r.setter = lambda ert, value : ert.enkf.site_config_set_lsf_request(ert.site_config, str(value))

    internalPanel.endPage()


    internalPanel.startPage("RSH")

    r = internalPanel.addRow(PathChooser(parent, "Command", "rsh_command", True))
    r.initialize = lambda ert : [ert.setTypes("site_config_get_rsh_command", ertwrapper.c_char_p),
                                 ert.setTypes("site_config_set_rsh_command", None, [ertwrapper.c_char_p])]
    r.getter = lambda ert : ert.enkf.site_config_get_rsh_command(ert.site_config)
    r.setter = lambda ert, value : ert.enkf.site_config_set_rsh_command(ert.site_config, str(value))

    r = internalPanel.addRow(IntegerSpinner(parent, "Max running", "max_running_rsh", 1, 1000))
    r.initialize = lambda ert : [ert.setTypes("site_config_get_max_running_rsh", ertwrapper.c_int),
                                 ert.setTypes("site_config_set_max_running_rsh", None, [ertwrapper.c_int])]
    r.getter = lambda ert : ert.enkf.site_config_get_max_running_rsh(ert.site_config)
    r.setter = lambda ert, value : ert.enkf.site_config_set_max_running_rsh(ert.site_config, value)


    r = internalPanel.addRow(KeywordTable(parent, "Host List", "rsh_host_list", "Host", "Number of jobs"))
    r.initialize = lambda ert : [ert.setTypes("site_config_get_rsh_host_list"),
                                 ert.setTypes("site_config_clear_rsh_host_list", None),
                                 ert.setTypes("site_config_add_rsh_host", None, [ertwrapper.c_char_p, ertwrapper.c_int])]
    r.getter = lambda ert : ert.getHash(ert.enkf.site_config_get_rsh_host_list(ert.site_config), True)

    def add_rsh_host(ert, listOfKeywords):
        ert.enkf.site_config_clear_rsh_host_list(ert.site_config)

        for keyword in listOfKeywords:
            if keyword[1].strip() == "":
                max_running = 1
            else:
                max_running = int(keyword[1])

            ert.enkf.site_config_add_rsh_host(ert.site_config, keyword[0], max_running)

    r.setter = add_rsh_host


    internalPanel.endPage()

    internalPanel.startPage("LOCAL")

    r = internalPanel.addRow(IntegerSpinner(parent, "Max running", "max_running_local", 1, 1000))
    r.initialize = lambda ert : [ert.setTypes("site_config_get_max_running_local", ertwrapper.c_int),
                                 ert.setTypes("site_config_set_max_running_local", None, [ertwrapper.c_int])]
    r.getter = lambda ert : ert.enkf.site_config_get_max_running_local(ert.site_config)
    r.setter = lambda ert, value : ert.enkf.site_config_set_max_running_local(ert.site_config, value)

    internalPanel.endPage()
    configPanel.addRow(internalPanel)

    configPanel.endPage()
