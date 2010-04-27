from PyQt4 import QtGui, QtCore
from widgets.configpanel import ConfigPanel
from widgets.pathchooser import PathChooser
from widgets.tablewidgets import KeywordTable
from widgets.tablewidgets import KeywordList
from widgets.stringbox import StringBox
import os
import ertwrapper

class EditJobDialog(QtGui.QDialog):
    """
    A panel for creating custom jobs.
    """
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.setModal(True)
        self.setWindowTitle("Edit job")
        self.setMinimumWidth(650)

        layout = QtGui.QVBoxLayout()

        self.jobPanel = JobConfigPanel(parent)

        layout.addWidget(self.jobPanel)

        self.doneButton = QtGui.QPushButton("Done", self)
        self.connect(self.doneButton, QtCore.SIGNAL('clicked()'), self.accept)

        buttonLayout = QtGui.QHBoxLayout()
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.doneButton)
        
        layout.addSpacing(10)
        layout.addLayout(buttonLayout)

        self.setLayout(layout)

    def closeEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        if not event.key() == QtCore.Qt.Key_Escape:
            QtGui.QDialog.keyPressEvent(self, event)

    def setRunningState(self, state):
        self.doneButton.setEnabled(not state)

    def setJob(self, job):
        self.jobPanel.setJob(job)



class JobConfigPanel(ConfigPanel):
    def initialize(self, ert):
        if not self.initialized:
            ert.setTypes("site_config_get_installed_jobs")
            ert.setTypes("ext_job_get_stdin_file", ertwrapper.c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_set_stdin_file", None, ertwrapper.c_char_p, library=ert.job_queue)
            ert.setTypes("ext_joblist_get_job", argtypes=ertwrapper.c_char_p, library=ert.job_queue)

            self.initialized = True

    def __init__(self, parent=None):
        ConfigPanel.__init__(self, parent)

        self.initialized = False

        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        def jid(ert):
            jl = ert.enkf.site_config_get_installed_jobs(ert.site_config)
            return ert.job_queue.ext_joblist_get_job(jl, self.job.name)

        self.stdin = PathChooser(self, "", "install_job_stdin", show_files=True, must_be_set=False)
        self.stdin.setter = lambda ert, value : ert.job_queue.ext_job_set_stdin_file(jid(ert), value)
        self.stdin.getter = lambda ert : ert.job_queue.ext_job_get_stdin_file(jid(ert))

#        self.stdout = PathChooser(self, "", "install_job_stdout", show_files=True, must_be_set=False, must_exist=False)
#        self.stdout.setter = lambda model, value: self.set("stdout", value)
#        self.stdout.getter = lambda model: self.jobsModel.stdout
#
#        self.stderr = PathChooser(self, "", "install_job_stderr", show_files=True, must_be_set=False, must_exist=False)
#        self.stderr.setter = lambda model, value: self.set("stderr", value)
#        self.stderr.getter = lambda model: self.jobsModel.stderr
#
#        self.target_file = PathChooser(self, "", "install_job_target_file", show_files=True, must_be_set=False,
#                                       must_exist=False)
#        self.target_file.setter = lambda model, value: self.set("target_file", value)
#        self.target_file.getter = lambda model: self.jobsModel.target_file
#
#        self.portable_exe = PathChooser(self, "", "install_job_portable_exe", show_files=True, must_be_set=False,
#                                        must_exist=False)
#        self.portable_exe.setter = lambda model, value: self.set("portable_exe", value)
#        self.portable_exe.getter = lambda model: self.jobsModel.portable_exe
#
#        self.platform_exe = KeywordTable(self, "", "install_job_platform_exe", colHead1="Platform",
#                                         colHead2="Executable")
#        self.platform_exe.setter = lambda model, value: self.set("platform_exe", value)
#        self.platform_exe.getter = lambda model: self.jobsModel.platform_exe
#
#        self.init_code = KeywordList(self, "", "install_job_init_code")
#        self.init_code.setter = lambda model, value: self.set("init_code", value)
#        self.init_code.getter = lambda model: self.jobsModel.init_code
#
#        self.env = KeywordTable(self, "", "install_job_env", colHead1="Variable", colHead2="Value")
#        self.env.setter = lambda model, value: self.set("env", value)
#        self.env.getter = lambda model: self.jobsModel.env
#
#        self.arglist = StringBox(self, "", "install_job_arglist")
#        self.arglist.setter = lambda model, value: self.set("arglist", value)
#        self.arglist.getter = lambda model: self.jobsModel.arglist
#
#        self.lsf_resources = StringBox(self, "", "install_job_lsf_resources")
#        self.lsf_resources.setter = lambda model, value: self.set("lsf_resources", value)
#        self.lsf_resources.getter = lambda model: self.jobsModel.lsf_resources


        self.startPage("Standard")
        self.add("Stdin:", self.stdin)
#        self.add("Stdout:", self.stdout)
#        self.add("Stderr:", self.stderr)
#        self.add("Target file:", self.target_file)
#        self.add("Portable exe.:", self.portable_exe)
#        self.add("Env.:", self.env)
#        self.endPage()
#
#        self.startPage("Advanced")
#        self.add("Platform exe.:", self.platform_exe)
#        self.add("Init code.:", self.init_code)
#        self.add("Arglist.:", self.arglist)
#        self.add("LSF resources:", self.lsf_resources)
        self.endPage()

    def add(self, label, widget):
        self.addRow(widget, label)


    def setJob(self, job):
        self.job = job

        self.initialize(self.stdin.getModel())

        self.stdin.fetchContent()
#        self.stdout.fetchContent()
#        self.stderr.fetchContent()
#        self.target_file.fetchContent()
#        self.portable_exe.fetchContent()
#        self.platform_exe.fetchContent()
#        self.init_code.fetchContent()
#        self.env.fetchContent()
#        self.arglist.fetchContent()
#        self.lsf_resources.fetchContent()
