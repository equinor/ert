# Some comments :)
from PyQt4 import QtGui, QtCore
import sys
import local
import os

import ertwrapper

from widgets.configpanel import ConfigPanel
from widgets.checkbox import CheckBox
from widgets.combochoice import ComboChoice
from widgets.pathchooser import PathChooser
from widgets.stringbox import StringBox
from widgets.helpedwidget import ContentModel, HelpedWidget
from widgets.tablewidgets import KeywordList, KeywordTable, MultiColumnTable
from widgets.spinnerwidgets import DoubleSpinner, IntegerSpinner
from pages.application import Application
import widgets.util

#for k in QtGui.QStyleFactory.keys():
#    print k
#
#QtGui.QApplication.setStyle("Plastique")
from pages.plotpanel import PlotPanel
from pages.parameters.parameterpanel import ParameterPanel

#todo: proper support for unicode characters?
from widgets.validateddialog import ValidatedDialog

app = QtGui.QApplication(sys.argv)

widget = Application()


site_config = "/project/res/etc/ERT/Config/site-config"
enkf_config = local.enkf_config
enkf_so     = local.enkf_so

ert = ertwrapper.ErtWrapper(site_config = site_config, enkf_config = enkf_config, enkf_so = enkf_so)



configPanel = ConfigPanel(widget)

# ----------------------------------------------------------------------------------------------
# Eclipse tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Eclipse")

#todo should be special % name type
r = configPanel.addRow(PathChooser(widget, "Eclipse Base", "eclbase"))
r.initialize = lambda ert : [ert.setTypes("ecl_config_get_eclbase", ertwrapper.c_char_p),
                             ert.setTypes("ecl_config_set_eclbase", None, ertwrapper.c_char_p)]
r.getter = lambda ert : ert.enkf.ecl_config_get_eclbase(ert.ecl_config)
r.setter = lambda ert, value : ert.enkf.ecl_config_set_eclbase(ert.ecl_config, str(value))

r = configPanel.addRow(PathChooser(widget, "Data file", "data_file"))
r.initialize = lambda ert : [ert.setTypes("ecl_config_get_data_file", ertwrapper.c_char_p),
                             ert.setTypes("ecl_config_set_data_file", None, ertwrapper.c_char_p)]
r.getter = lambda ert : ert.enkf.ecl_config_get_data_file(ert.ecl_config)
r.setter = lambda ert, value : ert.enkf.ecl_config_set_data_file(ert.ecl_config, str(value))

r = configPanel.addRow(PathChooser(widget, "Grid", "grid"))
r.initialize = lambda ert : [ert.setTypes("ecl_config_get_gridfile", ertwrapper.c_char_p),
                             ert.setTypes("ecl_config_set_grid", None, ertwrapper.c_char_p)]
r.getter = lambda ert : ert.enkf.ecl_config_get_gridfile(ert.ecl_config)
r.setter = lambda ert, value : ert.enkf.ecl_config_set_grid(ert.ecl_config, str(value))

r = configPanel.addRow(PathChooser(widget, "Schedule file" , "schedule_file" , files = True))
r.initialize = lambda ert : [ert.setTypes("ecl_config_get_schedule_file", ertwrapper.c_char_p),
                             ert.setTypes("ecl_config_set_schedule_file", None, ertwrapper.c_char_p)]
r.getter = lambda ert : ert.enkf.ecl_config_get_schedule_file(ert.ecl_config)
r.setter = lambda ert, value : ert.enkf.ecl_config_set_schedule_file(ert.ecl_config, str(value))


r = configPanel.addRow(PathChooser(widget, "Init section", "init_section"))
r.initialize = lambda ert : [ert.setTypes("ecl_config_get_init_section", ertwrapper.c_char_p),
                             ert.setTypes("ecl_config_set_init_section", None, ertwrapper.c_char_p)]
r.getter = lambda ert : ert.enkf.ecl_config_get_init_section(ert.ecl_config)
r.setter = lambda ert, value : ert.enkf.ecl_config_set_init_section(ert.ecl_config, str(value))


r = configPanel.addRow(PathChooser(widget, "Refcase", "refcase", True))
r.initialize = lambda ert : [ert.setTypes("ecl_config_get_refcase_name", ertwrapper.c_char_p),
                             ert.setTypes("ecl_config_set_refcase", None, ertwrapper.c_char_p)]
r.getter = lambda ert : ert.enkf.ecl_config_get_refcase_name(ert.ecl_config)
r.setter = lambda ert, value : ert.enkf.ecl_config_set_refcase(ert.ecl_config, str(value))

r = configPanel.addRow(PathChooser(widget, "Schedule prediction file", "schedule_prediction_file"))
r.getter = lambda ert : ert.getAttribute("schedule_prediction_file")
r.setter = lambda ert, value : ert.setAttribute("schedule_prediction_file", value)

r = configPanel.addRow(KeywordTable(widget, "Data keywords", "data_kw"))
r.initialize = lambda ert : [ert.setTypes("enkf_main_get_data_kw"),
                             ert.setTypes("enkf_main_clear_data_kw", None),
                             ert.setTypes("enkf_main_add_data_kw", None, [ertwrapper.c_char_p, ertwrapper.c_char_p])]
r.getter = lambda ert : ert.getSubstitutionList(ert.enkf.enkf_main_get_data_kw(ert.main))

def add_data_kw(ert, listOfKeywords):
    ert.enkf.enkf_main_clear_data_kw(ert.main)

    for keyword in listOfKeywords:
        ert.enkf.enkf_main_add_data_kw(ert.main, keyword[0], keyword[1])

r.setter = add_data_kw



configPanel.addSeparator()

internalPanel = ConfigPanel(widget)

internalPanel.startPage("Static keywords")

r = internalPanel.addRow(KeywordList(widget, "", "add_static_kw"))
r.initialize = lambda ert : [ert.setTypes("ecl_config_get_static_kw_list"),
                             ert.setTypes("ecl_config_clear_static_kw", None),
                             ert.setTypes("ecl_config_add_static_kw", None, [ertwrapper.c_char_p, ertwrapper.c_char_p])]
r.getter = lambda ert : ert.getStringList(ert.enkf.ecl_config_get_static_kw_list(ert.ecl_config))

def add_static_kw(ert, listOfKeywords):
    ert.enkf.ecl_config_clear_static_kw(ert.ecl_config)

    for keyword in listOfKeywords:
        ert.enkf.ecl_config_add_static_kw(ert.ecl_config, keyword)

r.setter = add_static_kw

internalPanel.endPage()

# todo: add support for fixed length schedule keywords
#internalPanel.startPage("Fixed length schedule keywords")
#
#r = internalPanel.addRow(KeywordList(widget, "", "add_fixed_length_schedule_kw"))
#r.getter = lambda ert : ert.getAttribute("add_fixed_length_schedule_kw")
#r.setter = lambda ert, value : ert.setAttribute("add_fixed_length_schedule_kw", value)
#
#internalPanel.endPage()

configPanel.addRow(internalPanel)

configPanel.endPage()


# ----------------------------------------------------------------------------------------------
# Analysis tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Analysis")

r = configPanel.addRow(CheckBox(widget, "ENKF rerun", "enkf_rerun", "Perform rerun"))
r.initialize = lambda ert : [ert.setTypes("analysis_config_get_rerun", ertwrapper.c_int),
                             ert.setTypes("analysis_config_set_rerun", None, [ertwrapper.c_int])]
r.getter = lambda ert : ert.enkf.analysis_config_get_rerun(ert.analysis_config)
r.setter = lambda ert, value : ert.enkf.analysis_config_set_rerun(ert.analysis_config, value)

r = configPanel.addRow(IntegerSpinner(widget, "Rerun start", "rerun_start",  0, 100000))
r.initialize = lambda ert : [ert.setTypes("analysis_config_get_rerun_start", ertwrapper.c_int),
                             ert.setTypes("analysis_config_set_rerun_start", None, [ertwrapper.c_int])]
r.getter = lambda ert : ert.enkf.analysis_config_get_rerun_start(ert.analysis_config)
r.setter = lambda ert, value : ert.enkf.analysis_config_set_rerun_start(ert.analysis_config, value)

r = configPanel.addRow(PathChooser(widget, "ENKF schedule file", "enkf_sched_file"))
r.initialize = lambda ert : [ert.setTypes("model_config_get_enkf_sched_file", ertwrapper.c_char_p),
                             ert.setTypes("enkf_main_get_model_config"),
                             ert.setTypes("model_config_set_enkf_sched_file", None, [ertwrapper.c_char_p])]
r.getter = lambda ert : ert.enkf.model_config_get_enkf_sched_file(ert.enkf.enkf_main_get_model_config(ert.main))
r.setter = lambda ert, value : ert.enkf.model_config_set_enkf_sched_file(ert.enkf.enkf_main_get_model_config(ert.main), str(value))

r = configPanel.addRow(KeywordList(widget, "Local config", "local_config"))
r.newKeywordPopup = lambda : QtGui.QFileDialog.getOpenFileName(r, "Select a path", "")
r.initialize = lambda ert : [ert.setTypes("local_config_get_config_files"),
                             ert.setTypes("enkf_main_get_local_config"),
                             ert.setTypes("local_config_clear_config_files", None),
                             ert.setTypes("local_config_add_config_file"), None, ertwrapper.c_char_p]

def get_local_config_files(ert):
    local_config = ert.enkf.enkf_main_get_local_config(ert.main)
    config_files_pointer = ert.enkf.local_config_get_config_files(local_config)
    return ert.getStringList(config_files_pointer)

r.getter = get_local_config_files

def add_config_file(ert, value):
    local_config = ert.enkf.enkf_main_get_local_config(ert.main)
    ert.enkf.local_config_clear_config_files(local_config)

    for file in value:
        ert.enkf.local_config_add_config_file(local_config, file)

r.setter = add_config_file

r = configPanel.addRow(PathChooser(widget, "Update log", "update_log"))
r.initialize = lambda ert : [ert.setTypes("analysis_config_get_log_path", ertwrapper.c_char_p),
                             ert.setTypes("analysis_config_set_log_path", None, [ertwrapper.c_char_p])]
r.getter = lambda ert : ert.enkf.analysis_config_get_log_path(ert.analysis_config)
r.setter = lambda ert, value : ert.enkf.analysis_config_set_log_path(ert.analysis_config, str(value))


configPanel.startGroup("EnKF")

r = configPanel.addRow(DoubleSpinner(widget, "Alpha", "enkf_alpha", 0, 100000, 2))
r.initialize = lambda ert : [ert.setTypes("analysis_config_get_alpha", ertwrapper.c_double),
                             ert.setTypes("analysis_config_set_alpha", None, [ertwrapper.c_double])]
r.getter = lambda ert : ert.enkf.analysis_config_get_alpha(ert.analysis_config)
r.setter = lambda ert, value : ert.enkf.analysis_config_set_alpha(ert.analysis_config, value)

r = configPanel.addRow(CheckBox(widget, "Merge Observations", "enkf_merge_observations", "Perform merge"))
r.initialize = lambda ert : [ert.setTypes("analysis_config_get_merge_observations", ertwrapper.c_int),
                             ert.setTypes("analysis_config_set_merge_observations", None, [ertwrapper.c_int])]
r.getter = lambda ert : ert.enkf.analysis_config_get_merge_observations(ert.analysis_config)
r.setter = lambda ert, value : ert.enkf.analysis_config_set_merge_observations(ert.analysis_config, value)


enkf_mode_type = {"ENKF_STANDARD" : 10, "ENKF_SQRT" : 20}
enkf_mode_type_inverted = {10 : "ENKF_STANDARD" , 20 : "ENKF_SQRT"}
r = configPanel.addRow(ComboChoice(widget, enkf_mode_type.keys(), "Mode", "enkf_mode"))
r.initialize = lambda ert : [ert.setTypes("analysis_config_get_enkf_mode", ertwrapper.c_int),
                             ert.setTypes("analysis_config_set_enkf_mode", None, [ertwrapper.c_int])]
r.getter = lambda ert : enkf_mode_type_inverted[ert.enkf.analysis_config_get_enkf_mode(ert.analysis_config)]
r.setter = lambda ert, value : ert.enkf.analysis_config_set_enkf_mode(ert.analysis_config, enkf_mode_type[str(value)])


r = configPanel.addRow(DoubleSpinner(widget, "Truncation", "enkf_truncation", 0, 1, 2))
r.initialize = lambda ert : [ert.setTypes("analysis_config_get_truncation", ertwrapper.c_double),
                             ert.setTypes("analysis_config_set_truncation", None, [ertwrapper.c_double])]
r.getter = lambda ert : ert.enkf.analysis_config_get_truncation(ert.analysis_config)
r.setter = lambda ert, value : ert.enkf.analysis_config_set_truncation(ert.analysis_config, value)



configPanel.endGroup()
configPanel.endPage()


# ----------------------------------------------------------------------------------------------
# Queue System tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Queue System")

r = configPanel.addRow(ComboChoice(widget, ["LSF", "RSH", "LOCAL"], "Queue system", "queue_system"))
r.initialize = lambda ert : [ert.setTypes("site_config_get_job_queue_name", ertwrapper.c_char_p),
                             ert.setTypes("site_config_set_job_queue", None, [ertwrapper.c_char_p])]
r.getter = lambda ert : ert.enkf.site_config_get_job_queue_name(ert.site_config)
r.setter = lambda ert, value : ert.enkf.site_config_set_job_queue(ert.site_config, str(value))

internalPanel = ConfigPanel(widget)

internalPanel.startPage("LSF")

r = internalPanel.addRow(ComboChoice(widget, ["NORMAL", "FAST_LOCAL", "SHORT"], "Mode", "lsf_queue"))
r.initialize = lambda ert : [ert.setTypes("site_config_get_lsf_queue", ertwrapper.c_char_p),
                             ert.setTypes("site_config_set_lsf_queue", None, [ertwrapper.c_char_p])]
r.getter = lambda ert : ert.enkf.site_config_get_lsf_queue(ert.site_config)
r.setter = lambda ert, value : ert.enkf.site_config_set_lsf_queue(ert.site_config, str(value))

r = internalPanel.addRow(IntegerSpinner(widget, "Max running", "max_running_lsf", 1, 1000))
r.initialize = lambda ert : [ert.setTypes("site_config_get_max_running_lsf", ertwrapper.c_int),
                             ert.setTypes("site_config_set_max_running_lsf", None, [ertwrapper.c_int])]
r.getter = lambda ert : ert.enkf.site_config_get_max_running_lsf(ert.site_config)
r.setter = lambda ert, value : ert.enkf.site_config_set_max_running_lsf(ert.site_config, value)

r = internalPanel.addRow(StringBox(widget, "Resources", "lsf_resources"))
r.initialize = lambda ert : [ert.setTypes("site_config_get_lsf_request", ertwrapper.c_char_p),
                             ert.setTypes("site_config_set_lsf_request", None, [ertwrapper.c_char_p])]
r.getter = lambda ert : ert.enkf.site_config_get_lsf_request(ert.site_config)
r.setter = lambda ert, value : ert.enkf.site_config_set_lsf_request(ert.site_config, str(value))

internalPanel.endPage()


internalPanel.startPage("RSH")

r = internalPanel.addRow(PathChooser(widget, "Command", "rsh_command", True))
r.initialize = lambda ert : [ert.setTypes("site_config_get_rsh_command", ertwrapper.c_char_p),
                             ert.setTypes("site_config_set_rsh_command", None, [ertwrapper.c_char_p])]
r.getter = lambda ert : ert.enkf.site_config_get_rsh_command(ert.site_config)
r.setter = lambda ert, value : ert.enkf.site_config_set_rsh_command(ert.site_config, str(value))

r = internalPanel.addRow(IntegerSpinner(widget, "Max running", "max_running_rsh", 1, 1000))
r.initialize = lambda ert : [ert.setTypes("site_config_get_max_running_rsh", ertwrapper.c_int),
                             ert.setTypes("site_config_set_max_running_rsh", None, [ertwrapper.c_int])]
r.getter = lambda ert : ert.enkf.site_config_get_max_running_rsh(ert.site_config)
r.setter = lambda ert, value : ert.enkf.site_config_set_max_running_rsh(ert.site_config, value)


r = internalPanel.addRow(KeywordTable(widget, "Host List", "rsh_host_list", "Host", "Number of jobs"))
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

r = internalPanel.addRow(IntegerSpinner(widget, "Max running", "max_running_local", 1, 1000))
r.initialize = lambda ert : [ert.setTypes("site_config_get_max_running_local", ertwrapper.c_int),
                             ert.setTypes("site_config_set_max_running_local", None, [ertwrapper.c_int])]
r.getter = lambda ert : ert.enkf.site_config_get_max_running_local(ert.site_config)
r.setter = lambda ert, value : ert.enkf.site_config_set_max_running_local(ert.site_config, value)

internalPanel.endPage()
configPanel.addRow(internalPanel)

configPanel.endPage()



# ----------------------------------------------------------------------------------------------
# System tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("System")

r = configPanel.addRow(PathChooser(widget, "Job script", "job_script", True))
r.getter = lambda ert : ert.getAttribute("job_script")
r.setter = lambda ert, value : ert.setAttribute("job_script", value)

internalPanel = ConfigPanel(widget)
internalPanel.startPage("setenv")

r = internalPanel.addRow(KeywordTable(widget, "", "setenv"))
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

r = internalPanel.addRow(KeywordTable(widget, "", "update_path"))
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

r = internalPanel.addRow(KeywordTable(widget, "", "install_job"))
r.getter = lambda ert : ert.getAttribute("install_job")
r.setter = lambda ert, value : ert.setAttribute("install_job", value)

internalPanel.endPage()
configPanel.addRow(internalPanel)

configPanel.endPage()


# ----------------------------------------------------------------------------------------------
# Plot tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Plot")

r = configPanel.addRow(PathChooser(widget, "Output path", "plot_path"))
r.initialize = lambda ert : [ert.setTypes("plot_config_get_path", ertwrapper.c_char_p),
                             ert.setTypes("plot_config_set_path", None, [ertwrapper.c_char_p])]
r.getter = lambda ert : ert.enkf.plot_config_get_path(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_path(ert.plot_config, str(value))

r = configPanel.addRow(ComboChoice(widget, ["PLPLOT", "TEXT"], "Driver", "plot_driver"))
r.initialize = lambda ert : [ert.setTypes("plot_config_get_driver", ertwrapper.c_char_p),
                             ert.setTypes("plot_config_set_driver", None, [ertwrapper.c_char_p])]
r.getter = lambda ert : ert.enkf.plot_config_get_driver(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_driver(ert.plot_config, str(value))

r = configPanel.addRow(IntegerSpinner(widget, "Errorbar max", "plot_errorbar_max", 1, 10000000))
r.initialize = lambda ert : [ert.setTypes("plot_config_get_errorbar_max", ertwrapper.c_int),
                             ert.setTypes("plot_config_set_errorbar_max", None, [ertwrapper.c_int])]
r.getter = lambda ert : ert.enkf.plot_config_get_errorbar_max(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_errorbar_max(ert.plot_config, value)

r = configPanel.addRow(IntegerSpinner(widget, "Width", "plot_width", 1, 10000))
r.initialize = lambda ert : [ert.setTypes("plot_config_get_width", ertwrapper.c_int),
                             ert.setTypes("plot_config_set_width", None, [ertwrapper.c_int])]
r.getter = lambda ert : ert.enkf.plot_config_get_width(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_width(ert.plot_config, value)

r = configPanel.addRow(IntegerSpinner(widget, "Height", "plot_height", 1, 10000))
r.initialize = lambda ert : [ert.setTypes("plot_config_get_height", ertwrapper.c_int),
                             ert.setTypes("plot_config_set_height", None, [ertwrapper.c_int])]
r.getter = lambda ert : ert.enkf.plot_config_get_height(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_height(ert.plot_config, value)

r = configPanel.addRow(PathChooser(widget, "Image Viewer", "image_viewer", True))
r.initialize = lambda ert : [ert.setTypes("plot_config_get_viewer", ertwrapper.c_char_p),
                             ert.setTypes("plot_config_set_viewer", None, [ertwrapper.c_char_p])]
r.getter = lambda ert : ert.enkf.plot_config_get_viewer(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_viewer(ert.plot_config, value)

r = configPanel.addRow(ComboChoice(widget, ["bmp", "jpg", "png", "tif"], "Image type", "image_type"))
r.initialize = lambda ert : [ert.setTypes("plot_config_get_image_type", ertwrapper.c_char_p),
                             ert.setTypes("plot_config_set_image_type", None, [ertwrapper.c_char_p])]
r.getter = lambda ert : ert.enkf.plot_config_get_image_type(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_image_type(ert.plot_config, str(value))


configPanel.endPage()


# ----------------------------------------------------------------------------------------------
# Ensemble tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Ensemble")

r = configPanel.addRow(IntegerSpinner(widget, "Number of realizations", "num_realizations", 1, 10000))
r.getter = lambda ert : ert.getAttribute("num_realizations")
r.setter = lambda ert, value : ert.setAttribute("num_realizations", value)

#r = configPanel.addRow(KeywordList(widget, "Summary", "summary"))
#r.getter = lambda ert : ert.getAttribute("summary")
#r.setter = lambda ert, value : ert.setAttribute("summary", value)


configPanel.startGroup("Parameters")
r = configPanel.addRow(ParameterPanel(widget, "", "parameters"))
r.getter = lambda ert : ert.getAttribute("summary")
r.setter = lambda ert, value : ert.setAttribute("summary", value)
configPanel.endGroup()


#internalPanel = ConfigPanel(widget)
#
#internalPanel.startPage("Fields")
#
#r = internalPanel.addRow(MultiColumnTable(widget, "Dynamic", "field_dynamic", ["Name", "Min", "Max"]))
#r.getter = lambda ert : ert.getAttribute("field_dynamic")
#r.setter = lambda ert, value : ert.setAttribute("field_dynamic", value)
##r.setDelegate(1, DoubleSpinBoxDelegate(widget))
##r.setDelegate(2, DoubleSpinBoxDelegate(widget))
#
#r = internalPanel.addRow(MultiColumnTable(widget, "Parameter", "field_parameter", ["Name", "Min", "Max", "Init", "Output", "Eclipse file", "Init files"]))
#r.getter = lambda ert : ert.getAttribute("field_parameter")
#r.setter = lambda ert, value : ert.setAttribute("field_parameter", value)
#
#r = internalPanel.addRow(MultiColumnTable(widget, "General", "field_general", ["Name", "Min", "Max", "Init", "Output", "Eclipse file", "Init files", "File generated by EnKF", "File loaded by EnKF"]))
#r.getter = lambda ert : ert.getAttribute("field_general")
#r.setter = lambda ert, value : ert.setAttribute("field_general", value)
#
#internalPanel.endPage()
#
#internalPanel.startPage("Gen")
#
#r = internalPanel.addRow(MultiColumnTable(widget, "Keyword", "gen_kw", ["Name", "Template", "Eclipse include", "Priors"]))
#r.getter = lambda ert : ert.getAttribute("gen_kw")
#r.setter = lambda ert, value : ert.setAttribute("gen_kw", value)
#
#r = internalPanel.addRow(MultiColumnTable(widget, "Data", "gen_data", ["Name", "Result file", "Input", "Output", "Eclipse file", "Init files"]))
#r.getter = lambda ert : ert.getAttribute("gen_data")
#r.setter = lambda ert, value : ert.setAttribute("gen_data", value)
#
#r = internalPanel.addRow(MultiColumnTable(widget, "Param", "gen_param", ["Name", "Input", "Output", "Eclipse file", "Init files", "Template"]))
#r.getter = lambda ert : ert.getAttribute("gen_param")
#r.setter = lambda ert, value : ert.setAttribute("gen_param", value)
#
#internalPanel.endPage()
#configPanel.addRow(internalPanel)

configPanel.endPage()


# ----------------------------------------------------------------------------------------------
# Observations tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Observations")

r = configPanel.addRow(ComboChoice(widget, ["REFCASE_SIMULATED", "REFCASE_HISTORY"], "History source", "history_source"))
r.getter = lambda ert : ert.getAttribute("history_source")
r.setter = lambda ert, value : ert.setAttribute("history_source", value)

r = configPanel.addRow(PathChooser(widget, "Observations config", "obs_config", True))
r.getter = lambda ert : ert.getAttribute("obs_config")
r.setter = lambda ert, value : ert.setAttribute("obs_config", value)

configPanel.endPage()


# ----------------------------------------------------------------------------------------------
# Simulations tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Simulations")


r = configPanel.addRow(IntegerSpinner(widget, "Max submit", "max_submit", 1, 10000))
r.getter = lambda ert : ert.getAttribute("max_submit")
r.setter = lambda ert, value : ert.setAttribute("max_submit", value)

r = configPanel.addRow(IntegerSpinner(widget, "Max resample", "max_resample", 1, 10000))
r.getter = lambda ert : ert.getAttribute("max_resample")
r.setter = lambda ert, value : ert.setAttribute("max_resample", value)

r = configPanel.addRow(KeywordTable(widget, "Forward model", "forward_model", "Job", "Arguments"))
r.getter = lambda ert : ert.getAttribute("forward_model")
r.setter = lambda ert, value : ert.setAttribute("forward_model", value)

r = configPanel.addRow(PathChooser(widget, "Case table", "case_table"))
r.getter = lambda ert : ert.getAttribute("case_table")
r.setter = lambda ert, value : ert.setAttribute("case_table", value)

r = configPanel.addRow(PathChooser(widget, "License path", "license_path"))
r.getter = lambda ert : ert.getAttribute("license_path")
r.setter = lambda ert, value : ert.setAttribute("license_path", value)


internalPanel = ConfigPanel(widget)

internalPanel.startPage("Runpath")

r = internalPanel.addRow(PathChooser(widget, "Runpath", "runpath"))
r.getter = lambda ert : ert.getAttribute("runpath")
r.setter = lambda ert, value : ert.setAttribute("runpath", value)

r = internalPanel.addRow(CheckBox(widget, "Pre clear", "pre_clear_runpath", "Perform pre clear"))
r.getter = lambda ert : ert.getAttribute("pre_clear_runpath")
r.setter = lambda ert, value : ert.setAttribute("pre_clear_runpath", value)

r = internalPanel.addRow(StringBox(widget, "Delete", "delete_runpath"))
r.getter = lambda ert : ert.getAttribute("delete_runpath")
r.setter = lambda ert, value : ert.setAttribute("delete_runpath", value)

r = internalPanel.addRow(StringBox(widget, "Keep", "keep_runpath"))
r.getter = lambda ert : ert.getAttribute("keep_runpath")
r.setter = lambda ert, value : ert.setAttribute("keep_runpath", value)

internalPanel.endPage()

internalPanel.startPage("Run Template")

r = internalPanel.addRow(MultiColumnTable(widget, "", "run_template", ["Template", "Target file", "Arguments"]))
r.getter = lambda ert : ert.getAttribute("run_template")
r.setter = lambda ert, value : ert.setAttribute("run_template", value)

#r = internalPanel.addRow(PathChooser(widget, "Template", "run_template", True))
#r.getter = lambda ert : ert.getAttribute("run_template")
#r.setter = lambda ert, value : ert.setAttribute("run_template", value)
#
#r = internalPanel.addRow(PathChooser(widget, "Target file", "target_file", True))
#r.getter = lambda ert : ert.getAttribute("target_file")
#r.setter = lambda ert, value : ert.setAttribute("target_file", value)
#
#r = internalPanel.addRow(KeywordTable(widget, "Arguments", "template_arguments"))
#r.getter = lambda ert : ert.getAttribute("template_arguments")
#r.setter = lambda ert, value : ert.setAttribute("template_arguments", value)

internalPanel.endPage()
configPanel.addRow(internalPanel)


configPanel.endPage()


# ----------------------------------------------------------------------------------------------
# dbase tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("dbase")

r = configPanel.addRow(ComboChoice(widget, ["BLOCK_FS", "PLAIN"], "dbase type", "dbase_type"))
r.getter = lambda ert : ert.getAttribute("dbase_type")
r.setter = lambda ert, value : ert.setAttribute("dbase_type", value)

r = configPanel.addRow(PathChooser(widget, "enspath", "enspath"))
r.getter = lambda ert : ert.getAttribute("enspath")
r.setter = lambda ert, value : ert.setAttribute("enspath", value)

configPanel.endPage()


# ----------------------------------------------------------------------------------------------
# Action tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Action")

r = configPanel.addRow(StringBox(widget, "Select case", "select_case"))
r.getter = lambda ert : ert.getAttribute("select_case")
r.setter = lambda ert, value : ert.setAttribute("select_case", value)

configPanel.endPage()


# ----------------------------------------------------------------------------------------------
# Log tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Log")

r = configPanel.addRow(PathChooser(widget, "Log file", "log_file", True))
r.getter = lambda ert : ert.getAttribute("log_file")
r.setter = lambda ert, value : ert.setAttribute("log_file", value)

r = configPanel.addRow(IntegerSpinner(widget, "Log level", "log_level", 0, 1000))
r.getter = lambda ert : ert.getAttribute("log_level")
r.setter = lambda ert, value : ert.setAttribute("log_level", value)

r = configPanel.addRow(PathChooser(widget, "Update log path", "update_log_path"))
r.getter = lambda ert : ert.getAttribute("update_log_path")
r.setter = lambda ert, value : ert.setAttribute("update_log_path", value)

configPanel.endPage()





widget.addPage("Configuration", widgets.util.resourceIcon("config"), configPanel)

initPanel = QtGui.QFrame()
initPanel.setFrameShape(QtGui.QFrame.Panel)
initPanel.setFrameShadow(QtGui.QFrame.Raised)

initPanelLayout = QtGui.QHBoxLayout()
initPanel.setLayout(initPanelLayout)

casePanel = QtGui.QFormLayout()

cases = KeywordList(widget, "", "case_list")

cases.newKeywordPopup = lambda list : ValidatedDialog(cases, "New case", "Enter name of new case:", list).showAndTell()
cases.addRemoveWidget.enableRemoveButton(False)
cases.list.setMaximumHeight(150)
cases.initialize = lambda ert : [ert.setTypes("enkf_main_get_fs"),
                                 ert.setTypes("enkf_fs_alloc_dirlist"),
                                 ert.setTypes("enkf_fs_has_dir", ertwrapper.c_int),
                                 ert.setTypes("enkf_fs_select_write_dir", None),
                                 ert.setTypes("enkf_fs_select_read_dir", None)]
def get_case_list(ert):
    fs = ert.enkf.enkf_main_get_fs(ert.main)
    caseList = ert.enkf.enkf_fs_alloc_dirlist(fs)

    list = ert.getStringList(caseList)
    ert.freeStringList(caseList)
    return list

def create_case(ert, cases):
    fs = ert.enkf.enkf_main_get_fs(ert.main)

    for case in cases:
        if not ert.enkf.enkf_fs_has_dir(fs, case):
            ert.enkf.enkf_fs_select_write_dir(fs, case, True)
            #ert.enkf.enkf_fs_select_read_dir(fs, case) #selection?
            break

cases.getter = get_case_list
cases.setter = create_case

casePanel.addRow("Cases:", cases)

initPanelLayout.addLayout(casePanel)


widget.addPage("Init", widgets.util.resourceIcon("db"), initPanel)

panel = QtGui.QFrame()
panel.setFrameShape(QtGui.QFrame.Panel)
panel.setFrameShadow(QtGui.QFrame.Raised)

panelLayout = QtGui.QVBoxLayout()
panel.setLayout(panelLayout)


import ctypes

def perform():
    fs = ert.enkf.enkf_main_get_fs(ert.main)
    caseList = ert.enkf.enkf_fs_alloc_dirlist(fs)

    list = ert.getStringList(caseList)
    print list
    ert.freeStringList(caseList)



button = QtGui.QPushButton("Get case list")
panel.connect(button, QtCore.SIGNAL('clicked()'), perform)

panelLayout.addWidget(button)

button = QtGui.QPushButton("Refetch")
panel.connect(button, QtCore.SIGNAL('clicked()'), ContentModel.updateObservers)

panelLayout.addWidget(button)


widget.addPage("Run", widgets.util.resourceIcon("run"), panel)



widget.addPage("Plots", widgets.util.resourceIcon("plot"), PlotPanel("plots/default"))


ContentModel.contentModel = ert
ContentModel.updateObservers()



sys.exit(app.exec_())




