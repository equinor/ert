# Some comments :)
from PyQt4 import QtGui, QtCore
import sys

import ertwrapper

from widgets.configpanel import ConfigPanel
from widgets.checkbox import CheckBox
from widgets.combochoice import ComboChoice
from widgets.pathchooser import PathChooser
from widgets.stringbox import StringBox
from widgets.helpedwidget import ContentModel, HelpedWidget
from widgets.tablewidgets import KeywordList, KeywordTable, MultiColumnTable
from widgets.spinnerwidgets import DoubleSpinner, IntegerSpinner

#for k in QtGui.QStyleFactory.keys():
#    print k
#
#QtGui.QApplication.setStyle("Plastique")

app = QtGui.QApplication(sys.argv)


widget = QtGui.QWidget()
widget.resize(750, 350)
widget.setWindowTitle('ERT GUI')

widgetLayout = QtGui.QVBoxLayout()


ert = ertwrapper.ErtWrapper()



configPanel = ConfigPanel(widget)

# ----------------------------------------------------------------------------------------------
# Eclipse tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Eclipse")

#todo should be special % name type
r = configPanel.addRow(PathChooser(widget, "Eclipse Base", "eclbase"))
r.getter = lambda ert : ert.getAttribute("eclbase")
r.setter = lambda ert, value : ert.setAttribute("eclbase", value)

r = configPanel.addRow(PathChooser(widget, "Data file", "data_file"))
r.getter = lambda ert : ert.getAttribute("data_file")
r.setter = lambda ert, value : ert.setAttribute("data_file", value)

r = configPanel.addRow(PathChooser(widget, "Grid", "grid"))
r.getter = lambda ert : ert.getAttribute("grid")
r.setter = lambda ert, value : ert.setAttribute("grid", value)

r = configPanel.addRow(PathChooser(widget, "Schedule file", "schedule_file"))
r.getter = lambda ert : ert.getAttribute("schedule_file")
r.setter = lambda ert, value : ert.setAttribute("schedule_file", value)

r = configPanel.addRow(PathChooser(widget, "Init section", "init_section"))
r.getter = lambda ert : ert.getAttribute("init_section")
r.setter = lambda ert, value : ert.setAttribute("init_section", value)

r = configPanel.addRow(PathChooser(widget, "Equil init file", "equil_init_file"))
r.getter = lambda ert : ert.getAttribute("equil_init_file")
r.setter = lambda ert, value : ert.setAttribute("equil_init_file", value)

r = configPanel.addRow(PathChooser(widget, "Refcase", "refcase"))
r.getter = lambda ert : ert.getAttribute("refcase")
r.setter = lambda ert, value : ert.setAttribute("refcase", value)

r = configPanel.addRow(PathChooser(widget, "Schedule prediction file", "schedule_prediction_file"))
r.getter = lambda ert : ert.getAttribute("schedule_prediction_file")
r.setter = lambda ert, value : ert.setAttribute("schedule_prediction_file", value)

r = configPanel.addRow(KeywordTable(widget, "Data keywords", "data_kw"))
r.getter = lambda ert : ert.getAttribute("data_kw")
r.setter = lambda ert, value : ert.setAttribute("data_kw", value)



#todo: add sepearator line

r = configPanel.addRow(QtGui.QFrame(widget))
r.setFrameShape(QtGui.QFrame.HLine)
r.setFrameShadow(QtGui.QFrame.Sunken)

internalPanel = ConfigPanel(widget)

internalPanel.startPage("Static keywords")

r = internalPanel.addRow(KeywordList(widget, "", "add_static_kw"))
r.getter = lambda ert : ert.getAttribute("add_static_kw")
r.setter = lambda ert, value : ert.setAttribute("add_static_kw", value)

internalPanel.endPage()

internalPanel.startPage("Fixed length schedule keywords")

r = internalPanel.addRow(KeywordList(widget, "", "add_fixed_length_schedule_kw"))
r.getter = lambda ert : ert.getAttribute("add_fixed_length_schedule_kw")
r.setter = lambda ert, value : ert.setAttribute("add_fixed_length_schedule_kw", value)

internalPanel.endPage()

#configPanel.endGroup()
configPanel.addRow(internalPanel)

configPanel.endPage()


# ----------------------------------------------------------------------------------------------
# Output tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Output")

r = configPanel.addRow(CheckBox(widget, "ENKF rerun", "enkf_rerun", "Perform rerun"))
r.getter = lambda ert : ert.getAttribute("enkf_rerun")
r.setter = lambda ert, value : ert.setAttribute("enkf_rerun", value)

r = configPanel.addRow(IntegerSpinner(widget, "Rerun start", "rerun_start",  0, 100000))
r.getter = lambda ert : ert.getAttribute("rerun_start")
r.setter = lambda ert, value : ert.setAttribute("rerun_start", value)

r = configPanel.addRow(PathChooser(widget, "ENKF schedule file", "enkf_sched_file"))
r.getter = lambda ert : ert.getAttribute("enkf_sched_file")
r.setter = lambda ert, value : ert.setAttribute("enkf_sched_file", value)

r = configPanel.addRow(PathChooser(widget, "Local config", "local_config"))
r.getter = lambda ert : ert.getAttribute("local_config")
r.setter = lambda ert, value : ert.setAttribute("local_config", value)


configPanel.startGroup("EnKF")

r = configPanel.addRow(DoubleSpinner(widget, "Alpha", "enkf_alpha", 0, 100000, 2))
r.getter = lambda ert : ert.getAttribute("enkf_alpha")
r.setter = lambda ert, value : ert.setAttribute("enkf_alpha", value)

r = configPanel.addRow(CheckBox(widget, "Merge Observations", "enkf_merge_observations", "Perform merge"))
r.getter = lambda ert : ert.getAttribute("enkf_merge_observations")
r.setter = lambda ert, value : ert.setAttribute("enkf_merge_observations", value)

r = configPanel.addRow(ComboChoice(widget, ["STANDARD", "SQRT"], "Mode", "enkf_mode"))
r.getter = lambda ert : ert.getAttribute("enkf_mode")
r.setter = lambda ert, value : ert.setAttribute("enkf_mode", value)

r = configPanel.addRow(DoubleSpinner(widget, "Truncation", "enkf_truncation", 0, 1, 2))
r.getter = lambda ert : ert.getAttribute("enkf_truncation")
r.setter = lambda ert, value : ert.setAttribute("enkf_truncation", value)


configPanel.endGroup()
configPanel.endPage()


# ----------------------------------------------------------------------------------------------
# Queue System tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Queue System")

r = configPanel.addRow(ComboChoice(widget, ["LSF", "RSH", "LOCAL"], "Queue system", "queue_system"))
r.getter = lambda ert : ert.getAttribute("queue_system")
r.setter = lambda ert, value : ert.setAttribute("queue_system", value)

internalPanel = ConfigPanel(widget)

internalPanel.startPage("LSF")
#configPanel.startGroup("LSF")

r = internalPanel.addRow(ComboChoice(widget, ["NORMAL", "FAST_LOCAL", "SHORT"], "Mode", "lsf_queue"))
r.getter = lambda ert : ert.getAttribute("lsf_queue")
r.setter = lambda ert, value : ert.setAttribute("lsf_queue", value)

r = internalPanel.addRow(IntegerSpinner(widget, "Max running", "max_running_lsf", 1, 1000))
r.getter = lambda ert : ert.getAttribute("max_running_lsf")
r.setter = lambda ert, value : ert.setAttribute("max_running_lsf", value)

r = internalPanel.addRow(StringBox(widget, "Resources", "lsf_resources"))
r.getter = lambda ert : ert.getAttribute("lsf_resources")
r.setter = lambda ert, value : ert.setAttribute("lsf_resources", value)

internalPanel.endPage()
#configPanel.endGroup()


internalPanel.startPage("RSH")
#configPanel.startGroup("RSH")

r = internalPanel.addRow(PathChooser(widget, "Command", "rsh_command", True))
r.getter = lambda ert : ert.getAttribute("rsh_command")
r.setter = lambda ert, value : ert.setAttribute("rsh_command", value)

r = internalPanel.addRow(IntegerSpinner(widget, "Max running", "max_running_rsh", 1, 1000))
r.getter = lambda ert : ert.getAttribute("max_running_rsh")
r.setter = lambda ert, value : ert.setAttribute("max_running_rsh", value)

r = internalPanel.addRow(KeywordTable(widget, "Host List", "rsh_host_list", "Host", "Number of jobs"))
r.getter = lambda ert : ert.getAttribute("rsh_host_list")
r.setter = lambda ert, value : ert.setAttribute("rsh_host_list", value)

internalPanel.endPage()
#configPanel.endGroup()

internalPanel.startPage("LOCAL")
#configPanel.startGroup("LOCAL")

r = internalPanel.addRow(IntegerSpinner(widget, "Max running", "max_running_local", 1, 1000))
r.getter = lambda ert : ert.getAttribute("max_running_local")
r.setter = lambda ert, value : ert.setAttribute("max_running_local", value)

internalPanel.endPage()
#configPanel.endGroup()
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
r.getter = lambda ert : ert.getAttribute("setenv")
r.setter = lambda ert, value : ert.setAttribute("setenv", value)

internalPanel.endPage()

internalPanel.startPage("Update path")

r = internalPanel.addRow(KeywordTable(widget, "", "update_path"))
r.getter = lambda ert : ert.getAttribute("update_path")
r.setter = lambda ert, value : ert.setAttribute("update_path", value)

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
r.initialize = lambda ert : ert.setRestype("plot_config_get_path", ertwrapper.c_char_p)
r.getter = lambda ert : ert.enkf.plot_config_get_path(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_path(ert.plot_config, str(value))

r = configPanel.addRow(ComboChoice(widget, ["PLPLOT", "TEXT"], "Driver", "plot_driver"))
r.initialize = lambda ert : ert.setRestype("plot_config_get_driver", ertwrapper.c_char_p)
r.getter = lambda ert : ert.enkf.plot_config_get_driver(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_driver(ert.plot_config, str(value))

r = configPanel.addRow(IntegerSpinner(widget, "Errorbar max", "plot_errorbar_max", 1, 10000000))
r.getter = lambda ert : ert.enkf.plot_config_get_errorbar_max(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_errorbar_max(ert.plot_config, value)

r = configPanel.addRow(IntegerSpinner(widget, "Width", "plot_width", 1, 10000))
r.getter = lambda ert : ert.enkf.plot_config_get_width(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_width(ert.plot_config, value)

r = configPanel.addRow(IntegerSpinner(widget, "Height", "plot_height", 1, 10000))
r.getter = lambda ert : ert.enkf.plot_config_get_height(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_height(ert.plot_config, value)

r = configPanel.addRow(PathChooser(widget, "Image Viewer", "image_viewer", True))
r.initialize = lambda ert : ert.setRestype("plot_config_get_viewer", ertwrapper.c_char_p)
r.getter = lambda ert : ert.enkf.plot_config_get_viewer(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_viewer(ert.plot_config, value)

r = configPanel.addRow(ComboChoice(widget, ["bmp", "jpg", "png", "tif"], "Image type", "image_type"))
r.initialize = lambda ert : ert.setRestype("plot_config_get_image_type", ertwrapper.c_char_p)
r.getter = lambda ert : ert.enkf.plot_config_get_image_type(ert.plot_config)
r.setter = lambda ert, value : ert.enkf.plot_config_set_image_type(ert.plot_config, str(value))


configPanel.endPage()


# ----------------------------------------------------------------------------------------------
# Ensemble tab
# ----------------------------------------------------------------------------------------------
configPanel.startPage("Ensemble")

r = configPanel.addRow(IntegerSpinner(widget, "# realizations", "num_realizations", 1, 10000))
r.getter = lambda ert : ert.getAttribute("num_realizations")
r.setter = lambda ert, value : ert.setAttribute("num_realizations", value)

r = configPanel.addRow(KeywordList(widget, "Summary", "summary"))
r.getter = lambda ert : ert.getAttribute("summary")
r.setter = lambda ert, value : ert.setAttribute("summary", value)

internalPanel = ConfigPanel(widget)

internalPanel.startPage("Fields")

r = internalPanel.addRow(MultiColumnTable(widget, "Dynamic", "field_dynamic", ["Name", "Min", "Max"]))
r.getter = lambda ert : ert.getAttribute("field_dynamic")
r.setter = lambda ert, value : ert.setAttribute("field_dynamic", value)

r = internalPanel.addRow(MultiColumnTable(widget, "Parameter", "field_parameter", ["Name", "Min", "Max", "Init", "Output", "Eclipse file", "Init files"]))
r.getter = lambda ert : ert.getAttribute("field_parameter")
r.setter = lambda ert, value : ert.setAttribute("field_parameter", value)

r = internalPanel.addRow(MultiColumnTable(widget, "General", "field_general", ["Name", "Min", "Max", "Init", "Output", "Eclipse file", "Init files", "File generated by EnKF", "File loaded by EnKF"]))
r.getter = lambda ert : ert.getAttribute("field_general")
r.setter = lambda ert, value : ert.setAttribute("field_general", value)

internalPanel.endPage()

internalPanel.startPage("Gen")

r = internalPanel.addRow(MultiColumnTable(widget, "Keyword", "gen_kw", ["Name", "Template", "Eclipse include", "Priors"]))
r.getter = lambda ert : ert.getAttribute("gen_kw")
r.setter = lambda ert, value : ert.setAttribute("gen_kw", value)

r = internalPanel.addRow(MultiColumnTable(widget, "Data", "gen_data", ["Name", "Result file", "Input", "Output", "Eclipse file", "Init files"]))
r.getter = lambda ert : ert.getAttribute("gen_data")
r.setter = lambda ert, value : ert.setAttribute("gen_data", value)

r = internalPanel.addRow(MultiColumnTable(widget, "Param", "gen_param", ["Name", "Input", "Output", "Eclipse file", "Init files", "Template"]))
r.getter = lambda ert : ert.getAttribute("gen_param")
r.setter = lambda ert, value : ert.setAttribute("gen_param", value)

internalPanel.endPage()
configPanel.addRow(internalPanel)

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

r = configPanel.addRow(PathChooser(widget, "License path", "license_path"))
r.getter = lambda ert : ert.getAttribute("license_path")
r.setter = lambda ert, value : ert.setAttribute("license_path", value)

r = configPanel.addRow(IntegerSpinner(widget, "Max submit", "max_submit", 1, 10000))
r.getter = lambda ert : ert.getAttribute("max_submit")
r.setter = lambda ert, value : ert.setAttribute("max_submit", value)

r = configPanel.addRow(IntegerSpinner(widget, "Max resample", "max_resample", 1, 10000))
r.getter = lambda ert : ert.getAttribute("max_resample")
r.setter = lambda ert, value : ert.setAttribute("max_resample", value)

r = configPanel.addRow(PathChooser(widget, "Case table", "case_table"))
r.getter = lambda ert : ert.getAttribute("case_table")
r.setter = lambda ert, value : ert.setAttribute("case_table", value)

r = configPanel.addRow(KeywordTable(widget, "Forward model", "forward_model", "Job", "Arguments"))
r.getter = lambda ert : ert.getAttribute("forward_model")
r.setter = lambda ert, value : ert.setAttribute("forward_model", value)



internalPanel = ConfigPanel(widget)

internalPanel.startPage("Runpath")
#configPanel.startGroup("Runpath")

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

#configPanel.endGroup()
internalPanel.endPage()

internalPanel.startPage("Run Template")
#configPanel.startGroup("Run Template")

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
#configPanel.endGroup()
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







ContentModel.contentModel = ert
ContentModel.updateObservers()



widgetLayout.addWidget(configPanel)

quitButton = QtGui.QPushButton("Close", widget)
widget.connect(quitButton, QtCore.SIGNAL('clicked()'), QtGui.qApp, QtCore.SLOT('quit()'))

buttonWidget = QtGui.QWidget(widget)
buttonLayout = QtGui.QHBoxLayout()
buttonLayout.addStretch(1)
buttonLayout.addWidget(quitButton)

buttonWidget.setLayout(buttonLayout)
widgetLayout.addWidget(buttonWidget)

widget.setLayout(widgetLayout)
widget.show()


sys.exit(app.exec_())