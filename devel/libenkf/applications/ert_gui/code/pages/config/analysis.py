# ----------------------------------------------------------------------------------------------
# Analysis tab
# ----------------------------------------------------------------------------------------------
from widgets.checkbox import CheckBox
import ertwrapper
from widgets.spinnerwidgets import IntegerSpinner, DoubleSpinner, DoubleSpinner
import widgets.tablewidgets
from widgets.pathchooser import PathChooser
from widgets.combochoice import ComboChoice
from PyQt4 import QtGui

def createAnalysisPage(configPanel, parent):
    configPanel.startPage("Analysis")

    r = configPanel.addRow(CheckBox(parent, "ENKF rerun", "config/analysis/enkf_rerun", "Perform rerun"))
    r.initialize = lambda ert : [ert.prototype("int analysis_config_get_rerun(long)"),
                                 ert.prototype("void analysis_config_set_rerun(long, bool)")]
    r.getter = lambda ert : ert.enkf.analysis_config_get_rerun(ert.analysis_config)
    r.setter = lambda ert, value : ert.enkf.analysis_config_set_rerun(ert.analysis_config, value)

    r = configPanel.addRow(IntegerSpinner(parent, "Rerun start", "config/analysis/rerun_start",  0, 100000))
    r.initialize = lambda ert : [ert.prototype("int analysis_config_get_rerun_start(long)"),
                                 ert.prototype("void analysis_config_set_rerun_start(long, int)")]
    r.getter = lambda ert : ert.enkf.analysis_config_get_rerun_start(ert.analysis_config)
    r.setter = lambda ert, value : ert.enkf.analysis_config_set_rerun_start(ert.analysis_config, value)

    r = configPanel.addRow(PathChooser(parent, "ENKF schedule file", "config/analysis/enkf_sched_file"))
    r.initialize = lambda ert : [ert.prototype("char* model_config_get_enkf_sched_file(long)"),
                                 ert.prototype("long enkf_main_get_model_config(long)"),
                                 ert.prototype("void model_config_set_enkf_sched_file(long, char*)")]
    r.getter = lambda ert : ert.enkf.model_config_get_enkf_sched_file(ert.enkf.enkf_main_get_model_config(ert.main))
    r.setter = lambda ert, value : ert.enkf.model_config_set_enkf_sched_file(ert.enkf.enkf_main_get_model_config(ert.main), str(value))

    r = configPanel.addRow(widgets.tablewidgets.KeywordList(parent, "Local config", "config/analysis/local_config"))
    r.newKeywordPopup = lambda list : QtGui.QFileDialog.getOpenFileName(r, "Select a path", "")
    r.initialize = lambda ert : [ert.prototype("long local_config_get_config_files(long)"),
                                 ert.prototype("long enkf_main_get_local_config(long)"),
                                 ert.prototype("void local_config_clear_config_files(long)"),
                                 ert.prototype("void local_config_add_config_file(long, char*)")]

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

    r = configPanel.addRow(PathChooser(parent, "Update log", "config/analysis/update_log"))
    r.initialize = lambda ert : [ert.prototype("char* analysis_config_get_log_path(long)"),
                                 ert.prototype("void analysis_config_set_log_path(long, char*)")]
    r.getter = lambda ert : ert.enkf.analysis_config_get_log_path(ert.analysis_config)
    r.setter = lambda ert, value : ert.enkf.analysis_config_set_log_path(ert.analysis_config, str(value))


    configPanel.startGroup("EnKF")

    r = configPanel.addRow(DoubleSpinner(parent, "Alpha", "config/analysis/enkf_alpha", 0, 100000, 2))
    r.initialize = lambda ert : [ert.prototype("double analysis_config_get_alpha(long)"),
                                 ert.prototype("void analysis_config_set_alpha(long, double)")]
    r.getter = lambda ert : ert.enkf.analysis_config_get_alpha(ert.analysis_config)
    r.setter = lambda ert, value : ert.enkf.analysis_config_set_alpha(ert.analysis_config, value)

    r = configPanel.addRow(CheckBox(parent, "Merge Observations", "config/analysis/enkf_merge_observations", "Perform merge"))
    r.initialize = lambda ert : [ert.prototype("bool analysis_config_get_merge_observations(long)"),
                                 ert.prototype("void analysis_config_set_merge_observations(long, int)")]
    r.getter = lambda ert : ert.enkf.analysis_config_get_merge_observations(ert.analysis_config)
    r.setter = lambda ert, value : ert.enkf.analysis_config_set_merge_observations(ert.analysis_config, value)


    enkf_mode_type = {"ENKF_STANDARD" : 10, "ENKF_SQRT" : 20}
    enkf_mode_type_inverted = {10 : "ENKF_STANDARD" , 20 : "ENKF_SQRT"}
    r = configPanel.addRow(ComboChoice(parent, enkf_mode_type.keys(), "Mode", "config/analysis/enkf_mode"))
    r.initialize = lambda ert : [ert.prototype("int analysis_config_get_enkf_mode(long)"),
                                 ert.prototype("void analysis_config_set_enkf_mode(long, int)")]
    r.getter = lambda ert : enkf_mode_type_inverted[ert.enkf.analysis_config_get_enkf_mode(ert.analysis_config)]
    r.setter = lambda ert, value : ert.enkf.analysis_config_set_enkf_mode(ert.analysis_config, enkf_mode_type[str(value)])


    r = configPanel.addRow(DoubleSpinner(parent, "Truncation", "config/analysis/enkf_truncation", 0, 1, 2))
    r.initialize = lambda ert : [ert.prototype("double analysis_config_get_truncation(long)"),
                                 ert.prototype("void analysis_config_set_truncation(long, double)")]
    r.getter = lambda ert : ert.enkf.analysis_config_get_truncation(ert.analysis_config)
    r.setter = lambda ert, value : ert.enkf.analysis_config_set_truncation(ert.analysis_config, value)



    configPanel.endGroup()
    configPanel.endPage()