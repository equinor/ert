# ----------------------------------------------------------------------------------------------
# Simulations tab
# ----------------------------------------------------------------------------------------------
from PyQt4 import QtCore
from widgets.spinnerwidgets import IntegerSpinner
import ertwrapper
from widgets.tablewidgets import KeywordTable, MultiColumnTable, MultiColumnTable
from widgets.pathchooser import PathChooser
from widgets.checkbox import CheckBox
from widgets.configpanel import ConfigPanel
from widgets.stringbox import StringBox
from pages.config.jobs.forwardmodelpanel import ForwardModelPanel

def createSimulationsPage(configPanel, parent):
    configPanel.startPage("Simulations")


    r = configPanel.addRow(IntegerSpinner(parent, "Max submit", "max_submit", 1, 10000))
    r.initialize = lambda ert : [ert.setTypes("site_config_get_max_submit", ertwrapper.c_int),
                                 ert.setTypes("site_config_set_max_submit", None, [ertwrapper.c_int])]
    r.getter = lambda ert : ert.enkf.site_config_get_max_submit(ert.site_config)
    r.setter = lambda ert, value : ert.enkf.site_config_set_max_submit(ert.site_config, value)

    r = configPanel.addRow(IntegerSpinner(parent, "Max resample", "max_resample", 1, 10000))
    r.initialize = lambda ert : [ert.setTypes("model_config_get_max_resample", ertwrapper.c_int),
                                 ert.setTypes("model_config_set_max_resample", None, [ertwrapper.c_int])]
    r.getter = lambda ert : ert.enkf.model_config_get_max_resample(ert.model_config)
    r.setter = lambda ert, value : ert.enkf.model_config_set_max_resample(ert.model_config, value)



    r = configPanel.addRow(ForwardModelPanel(parent))
    r.initialize = lambda ert: [ert.prototype("long model_config_get_forward_model(long)"),
                                ert.prototype("long site_config_get_installed_jobs(long)"),
                                ert.prototype("long ext_joblist_alloc_list(long)", lib=ert.job_queue),
                                ert.prototype("char* ext_job_get_private_args_as_string(long)", lib=ert.job_queue),
                                ert.prototype("char* ext_job_get_help_text(long)", lib=ert.job_queue),
                                ert.prototype("void forward_model_clear(long)", lib=ert.job_queue),
                                ert.prototype("long forward_model_add_job(long, char*)", lib=ert.job_queue),
                                ert.prototype("void ext_job_set_private_args_from_string(long, char*)", lib=ert.job_queue),
                                ert.prototype("long forward_model_alloc_joblist(long)", lib=ert.job_queue),]

    def get_forward_model(ert):
        site_config = ert.site_config
        installed_jobs_pointer = ert.enkf.site_config_get_installed_jobs(site_config)
        installed_jobs_stringlist_pointer = ert.job_queue.ext_joblist_alloc_list(installed_jobs_pointer)
        available_jobs = ert.getStringList(installed_jobs_stringlist_pointer, free_after_use=True)

        result = {'available_jobs': available_jobs}

        model_config = ert.model_config
        forward_model = ert.enkf.model_config_get_forward_model(model_config)
        name_string_list = ert.job_queue.forward_model_alloc_joblist(forward_model)
        job_names = ert.getStringList(name_string_list, free_after_use=True)

        forward_model_jobs = []

        count = 0
        for name in job_names:
            ext_job = ert.job_queue.forward_model_iget_job(forward_model, count)
            arg_string = ert.job_queue.ext_job_get_private_args_as_string(ext_job)
            help_text = ert.job_queue.ext_job_get_help_text(ext_job)
            forward_model_jobs.append((name, arg_string, help_text))
            count+=1

        result['forward_model'] = forward_model_jobs

        return result

    r.getter = get_forward_model

    def update_forward_model(ert, forward_model):
        forward_model_pointer = ert.enkf.model_config_get_forward_model(ert.model_config)
        ert.job_queue.forward_model_clear(forward_model_pointer)

        for job in forward_model:
            name = job[0]
            args = job[1]
            ext_job = ert.job_queue.forward_model_add_job(forward_model_pointer, name)
            ert.job_queue.ext_job_set_private_args_from_string(ext_job, args)

    r.setter = update_forward_model



    r = configPanel.addRow(PathChooser(parent, "Case table", "case_table"))
    r.getter = lambda ert : ert.getAttribute("case_table")
    r.setter = lambda ert, value : ert.setAttribute("case_table", value)

    r = configPanel.addRow(PathChooser(parent, "License path", "license_path"))
    r.initialize = lambda ert : [ert.setTypes("site_config_get_license_root_path__", ertwrapper.c_char_p),
                                 ert.setTypes("site_config_set_license_root_path", None, ertwrapper.c_char_p)]
    r.getter = lambda ert : ert.enkf.site_config_get_license_root_path__(ert.site_config)

    def ls(string):
        if string is None:
            return ""
        else:
            return string

    r.setter = lambda ert, value : ert.enkf.site_config_set_license_root_path(ert.site_config, ls(value))



    internalPanel = ConfigPanel(parent)

    internalPanel.startPage("Runpath")

    r = internalPanel.addRow(PathChooser(parent, "Runpath", "runpath", path_format=True))
    r.initialize = lambda ert : [ert.setTypes("model_config_get_runpath_as_char", ertwrapper.c_char_p),
                                 ert.setTypes("model_config_set_runpath_fmt", None, [ertwrapper.c_char_p])]
    r.getter = lambda ert : ert.enkf.model_config_get_runpath_as_char(ert.model_config)
    r.setter = lambda ert, value : ert.enkf.model_config_set_runpath_fmt(ert.model_config, str(value))
    parent.connect(r, QtCore.SIGNAL("contentsChanged()"), lambda : r.modelEmit("runpathChanged()"))

    r = internalPanel.addRow(CheckBox(parent, "Pre clear", "pre_clear_runpath", "Perform pre clear"))
    r.getter = lambda ert : ert.getAttribute("pre_clear_runpath")
    r.setter = lambda ert, value : ert.setAttribute("pre_clear_runpath", value)

    r = internalPanel.addRow(StringBox(parent, "Delete", "delete_runpath"))
    r.getter = lambda ert : ert.getAttribute("delete_runpath")
    r.setter = lambda ert, value : ert.setAttribute("delete_runpath", value)

    r = internalPanel.addRow(StringBox(parent, "Keep", "keep_runpath"))
    r.getter = lambda ert : ert.getAttribute("keep_runpath")
    r.setter = lambda ert, value : ert.setAttribute("keep_runpath", value)

    internalPanel.endPage()

    internalPanel.startPage("Run Template")

    r = internalPanel.addRow(MultiColumnTable(parent, "", "run_template", ["Template", "Target file", "Arguments"]))
    r.getter = lambda ert : ert.getAttribute("run_template")
    r.setter = lambda ert, value : ert.setAttribute("run_template", value)

    internalPanel.endPage()
    configPanel.addRow(internalPanel)


    configPanel.endPage()
