from PyQt4 import QtGui, QtCore
from widgets.configpanel import ConfigPanel
from widgets.pathchooser import PathChooser
from widgets.tablewidgets import KeywordTable
from widgets.tablewidgets import KeywordList
from widgets.stringbox import StringBox
import os
from ertwrapper import c_char_p, c_int
from widgets.spinnerwidgets import IntegerSpinner
import widgets.util
from widgets.helpedwidget import ContentModelProxy

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
        self.cancelButton = QtGui.QPushButton("Cancel", self)
        self.connect(self.doneButton, QtCore.SIGNAL('clicked()'), self.saveJob)
        self.connect(self.cancelButton, QtCore.SIGNAL('clicked()'), self.reject)

        self.validationInfo = widgets.util.ValidationInfo()

        buttonLayout = QtGui.QHBoxLayout()
        buttonLayout.addWidget(self.validationInfo)
        buttonLayout.addStretch(1)
        buttonLayout.addWidget(self.doneButton)
        buttonLayout.addWidget(self.cancelButton)

        layout.addSpacing(10)
        layout.addLayout(buttonLayout)

        self.setLayout(layout)


    def keyPressEvent(self, event):
        if not event.key() == QtCore.Qt.Key_Escape:
            QtGui.QDialog.keyPressEvent(self, event)

    def setJob(self, job):
        self.jobPanel.setJob(job)

    def saveJob(self):
        msg = self.jobPanel.saveJob()
        if msg is None:
            self.accept()
        else:
            self.validationInfo.setMessage(msg)


class JobConfigPanel(ConfigPanel):
    def __init__(self, parent=None):
        ConfigPanel.__init__(self, parent)

        self.initialized = False

        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        def jid(ert):
            """Returns the pointer to the current job (self.job)"""
            jl = ert.enkf.site_config_get_installed_jobs(ert.site_config)
            return ert.job_queue.ext_joblist_get_job(jl, self.job.name)

        self.stdin = PathChooser(self, "", "install_job_stdin", show_files=True, must_be_set=False, must_exist=True)
        self.stdin.setter = lambda ert, value : ert.job_queue.ext_job_set_stdin_file(jid(ert), value)
        self.stdin.getter = lambda ert : ert.job_queue.ext_job_get_stdin_file(jid(ert))

        self.stdout = PathChooser(self, "", "install_job_stdout", show_files=True, must_be_set=True, must_exist=False)
        self.stdout.setter = lambda ert, value : ert.job_queue.ext_job_set_stdout_file(jid(ert), value)
        self.stdout.getter = lambda ert : ert.job_queue.ext_job_get_stdout_file(jid(ert))

        self.stderr = PathChooser(self, "", "install_job_stderr", show_files=True, must_be_set=True, must_exist=False)
        self.stderr.setter = lambda ert, value : ert.job_queue.ext_job_set_stderr_file(jid(ert), value)
        self.stderr.getter = lambda ert : ert.job_queue.ext_job_get_stderr_file(jid(ert))

        self.target_file = PathChooser(self, "", "install_job_target_file", show_files=True, must_be_set=False,
                                       must_exist=False)
        self.target_file.setter = lambda ert, value : ert.job_queue.ext_job_set_target_file(jid(ert), value)
        self.target_file.getter = lambda ert : ert.job_queue.ext_job_get_target_file(jid(ert))

        self.executable = PathChooser(self, "", "install_job_executable", show_files=True, must_be_set=True,
                                      must_exist=True, is_executable_file=True)
        self.executable.setter = lambda ert, value : ert.job_queue.ext_job_set_executable(jid(ert), value)
        self.executable.getter = lambda ert : ert.job_queue.ext_job_get_executable(jid(ert))

        def setEnv(ert, value):
            job = jid(ert)
            ert.job_queue.ext_job_clear_environment(job)

            for env in value:
                ert.job_queue.ext_job_add_environment(job, env[0], env[1])

        self.env = KeywordTable(self, "", "install_job_env", colHead1="Variable", colHead2="Value")
        self.env.setter = setEnv
        self.env.getter = lambda ert : ert.getHash(ert.job_queue.ext_job_get_environment(jid(ert)))

        self.arglist = StringBox(self, "", "install_job_arglist")
        #self.arglist = KeywordList(self, "", "install_job_arglist")
        #self.arglist.setPopupLabels("New argument", "Enter name of new argument:")
        def set_arglist(ert, value):
            if not value is None:
                list_from_string = value.split(' ')
                #todo: missing setter
                print list_from_string

        self.arglist.setter = set_arglist

        def get_arglist(ert):
            arglist = ert.getStringList(ert.job_queue.ext_job_get_arglist(jid(ert)))
            string_from_list = " ".join(arglist)
            return string_from_list

        self.arglist.getter = get_arglist

        self.lsf_resources = StringBox(self, "", "install_job_lsf_resources")
        self.lsf_resources.setter = lambda ert, value : ert.job_queue.ext_job_set_lsf_request(jid(ert), value)
        self.lsf_resources.getter = lambda ert : ert.job_queue.ext_job_get_lsf_request(jid(ert))

        self.max_running = IntegerSpinner(self, "", "install_job_max_running", 0, 10000)
        self.max_running.setter = lambda ert, value : ert.job_queue.ext_job_set_max_running(jid(ert), value)
        self.max_running.getter = lambda ert : ert.job_queue.ext_job_get_max_running(jid(ert))

        self.max_running_minutes = IntegerSpinner(self, "", "install_job_max_running_minutes", 0, 10000)
        self.max_running_minutes.setter = lambda ert, value : ert.job_queue.ext_job_set_max_running_minutes(jid(ert), value)
        self.max_running_minutes.getter = lambda ert : ert.job_queue.ext_job_get_max_running_minutes(jid(ert))


        self.startPage("Standard")
        self.add("Stdin:", self.stdin)
        self.add("Stdout:", self.stdout)
        self.add("Stderr:", self.stderr)
        self.add("Target file:", self.target_file)
        self.add("Executable.:", self.executable)
        self.add("Env.:", self.env)
        self.endPage()

        self.startPage("Advanced")
        self.add("Arglist.:", self.arglist)
        self.add("LSF resources:", self.lsf_resources)
        self.add("Max running:", self.max_running)
        self.max_running.setInfo("(0=unlimited)")
        self.add("Max running minutes:", self.max_running_minutes)
        self.max_running_minutes.setInfo("(0=unlimited)")
        self.endPage()

    def add(self, label, widget):
        self.addRow(widget, label)


    def initialize(self, ert):
        if not self.initialized:
            ert.setTypes("site_config_get_installed_jobs")
            ert.setTypes("ext_job_get_stdin_file", c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_set_stdin_file", None, c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_get_stdout_file", c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_set_stdout_file", None, c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_get_stderr_file", c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_set_stderr_file", None, c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_get_target_file", c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_set_target_file", None, c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_get_executable", c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_set_executable", None, c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_get_lsf_request", c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_set_lsf_request", None, c_char_p, library=ert.job_queue)
            ert.setTypes("ext_job_get_max_running", c_int, library=ert.job_queue)
            ert.setTypes("ext_job_set_max_running", None, c_int, library=ert.job_queue)
            ert.setTypes("ext_job_get_max_running_minutes", c_int, library=ert.job_queue)
            ert.setTypes("ext_job_set_max_running_minutes", None, c_int, library=ert.job_queue)
            ert.setTypes("ext_job_get_environment", library=ert.job_queue)
            ert.setTypes("ext_job_add_environment", None, [c_char_p, c_char_p], library=ert.job_queue)
            ert.setTypes("ext_job_clear_environment", None, library=ert.job_queue)
            ert.setTypes("ext_job_save", None, library=ert.job_queue)
            ert.setTypes("ext_joblist_get_job", argtypes=c_char_p, library=ert.job_queue)

            self.initialized = True


    def setJob(self, job):
        self.job = job

        self.initialize(self.stdin.getModel())

        self.cmproxy = ContentModelProxy() #Since only the last change matters and no insert and remove is done
        self.cmproxy.proxify(self.stdin, self.stdout, self.stderr, self.target_file, self.executable,
                             self.env, self.arglist, self.lsf_resources, self.max_running, self.max_running_minutes)

        self.stdin.fetchContent()
        self.stdout.fetchContent()
        self.stderr.fetchContent()
        self.target_file.fetchContent()
        self.executable.fetchContent()
        self.env.fetchContent()
        self.arglist.fetchContent()
        self.lsf_resources.fetchContent()
        self.max_running.fetchContent()
        self.max_running_minutes.fetchContent()

    def saveJob(self):
        if self.executable.isValid() and self.stderr.isValid() and self.stdout.isValid():
            self.cmproxy.apply()

            ert = self.stdin.getModel()
            jl = ert.enkf.site_config_get_installed_jobs(ert.site_config)
            jid = ert.job_queue.ext_joblist_get_job(jl, self.job.name)
            ert.job_queue.ext_job_save(jid)
            return None
        else:
            return "These fields are required: executable, stdout and stderr!"

