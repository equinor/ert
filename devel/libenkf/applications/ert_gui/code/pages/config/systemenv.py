#----------------------------------------------------------------------------------------------
# System tab
# ----------------------------------------------------------------------------------------------
from widgets.pathchooser import PathChooser
from widgets.configpanel import ConfigPanel
from widgets.tablewidgets import KeywordTable
import ertwrapper
from widgets.helpedwidget import HelpedWidget
from widgets.searchablelist import SearchableList
from PyQt4 import QtGui, QtCore
import widgets.stringbox
import widgets.stringbox
import widgets.combochoice
import widgets.combochoice
from widgets.stringbox import StringBox
import widgets.validateddialog

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
                                 ert.setTypes("site_config_update_pathvar", None,
                                              [ertwrapper.c_char_p, ertwrapper.c_char_p])]
    def get_update_path(ert):
        paths = ert.getStringList(ert.enkf.site_config_get_path_variables(ert.site_config))
        values =  ert.getStringList(ert.enkf.site_config_get_path_values(ert.site_config))

        return [[p, v] for p, v in zip(paths, values)]

    r.getter = get_update_path

    def update_pathvar(ert, value):
        ert.enkf.site_config_clear_pathvar(ert.site_config)

        for pathvar in value:
            ert.enkf.site_config_update_pathvar(ert.site_config, pathvar[0], pathvar[1])

    r.setter = update_pathvar

    internalPanel.endPage()

    internalPanel.startPage("Install job")

    r = internalPanel.addRow(InstallJobsPanel(parent))
    r.initialize = lambda ert : [ert.setTypes("site_config_get_installed_jobs"),
                                 ert.setTypes("ext_job_is_private", ertwrapper.c_int, library=ert.job_queue),
                                 ert.setTypes("ext_joblist_get_jobs", library=ert.job_queue)]
    def get_jobs(ert):
        jl = ert.enkf.site_config_get_installed_jobs(ert.site_config)
        h  = ert.job_queue.ext_joblist_get_jobs(jl)

        jobs = ert.getHash(h)

        private_jobs = []
        for k, v in jobs:
            print k, v
            #if ert.job_queue.ext_job_is_private(v):
            #    private_jobs.append(k)

        print private_jobs

        #Itererer over hash_tabellen:
#        python_joblist = []
#        job = hash_get( h , job_name );
#        if ext_job_is_private( job ):                    <- Denne funksjonen er i libjob_queue
#            python_joblist.append( job_name )
#
    r.getter = get_jobs
    r.setter = lambda ert, value : ert.setAttribute("install_job", value)

    internalPanel.endPage()
    configPanel.addRow(internalPanel)

    configPanel.endPage()


class InstallJobsPanel(HelpedWidget):
    def __init__(self, parent=None):
        HelpedWidget.__init__(self, parent, "", "install_jobs")

        self.searchableList = SearchableList(parent, listHeight=200)
        self.addWidget(self.searchableList)

        self.connect(self.searchableList, QtCore.SIGNAL('currentItemChanged(QListWidgetItem, QListWidgetItem)'),
                     self.changeParameter)
        self.connect(self.searchableList, QtCore.SIGNAL('addItem(list)'), self.addItem)
        self.connect(self.searchableList, QtCore.SIGNAL('removeItem(list)'), self.removeItem)

        self.jobPanel = JobInfoPanel(parent)

        self.pagesWidget = QtGui.QStackedWidget()

        self.emptyPanel = QtGui.QFrame(self)

        self.emptyPanel.setFrameShape(QtGui.QFrame.StyledPanel)
        self.emptyPanel.setFrameShadow(QtGui.QFrame.Plain)
        self.emptyPanel.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.pagesWidget.addWidget(self.emptyPanel)
        self.pagesWidget.addWidget(self.jobPanel)
        self.addWidget(self.pagesWidget)

    def contentsChanged(self):
        """Called whenever the contents changes."""
        #self.updateContent(self.doubleBox.text().toDouble()[0])
        pass

    def fetchContent(self):
        """Retrieves data from the model and inserts it into the widget"""
        jobs = self.getFromModel()

    def changeParameter(self, current, previous):
        """Switch between jobs. Selection from the list"""
        if current is None:
            self.pagesWidget.setCurrentWidget(self.emptyPanel)
        else:
            self.pagesWidget.setCurrentWidget(self.jobPanel)
            self.jobPanel.setJobModel(current.data(QtCore.Qt.UserRole).toPyObject())

    def addToList(self, list, name):
        """Adds a new job to the list"""
        param = QtGui.QListWidgetItem()
        param.setText(name)

        param.setData(QtCore.Qt.UserRole, JobsModel(name))

        list.addItem(param)
        list.setCurrentItem(param)
        return param


    def addItem(self, list):
        """Called by the add button to insert a new job"""
        uniqueNames = []
        for index in range(list.count()):
            uniqueNames.append(str(list.item(index).text()))

        pd = widgets.validateddialog.ValidatedDialog(self, "New job", "Enter name of new job:", uniqueNames)
        if pd.exec_():
            self.addToList(list, pd.getName())

        #self.contentsChanged()
        #todo: tell forward model that a new variable is available


    def removeItem(self, list):
        """Called by the remove button to remove a selected parameter"""
        currentRow = list.currentRow()

        if currentRow >= 0:
            doDelete = QtGui.QMessageBox.question(self, "Delete job?", "Are you sure you want to delete the job?",
                                                  QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

            if doDelete == QtGui.QMessageBox.Yes:
                list.takeItem(currentRow)
            #self.contentsChanged()


class JobsModel:
    def __init__(self, name):
        self.name = name
        self.stdin = ""
        self.stdout = ""
        self.stderr = ""
        self.target_file = ""
        self.portable_exe = ""
        self.platform_exe = ""
        self.init_code = ""
        self.env = ""


class JobInfoPanel(QtGui.QFrame):
    def __init__(self, parent):
        QtGui.QFrame.__init__(self, parent)

        self.jobsModel = JobsModel("")

        self.setFrameShape(QtGui.QFrame.StyledPanel)
        self.setFrameShadow(QtGui.QFrame.Plain)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        self.stdin = PathChooser(self, "", "install_job_stdin", show_files=True, must_be_set=False)
        self.stdin.setter = lambda model, value: setattr(self.jobsModel, "stdin", value)
        self.stdin.getter = lambda model: self.jobsModel.stdin

        self.stdout = PathChooser(self, "", "install_job_stdout", show_files=True, must_be_set=False, must_exist=False)
        self.stdout.setter = lambda model, value: setattr(self.jobsModel, "stdout", value)
        self.stdout.getter = lambda model: self.jobsModel.stdout

        self.stderr = PathChooser(self, "", "install_job_stderr", show_files=True, must_be_set=False, must_exist=False)
        self.stderr.setter = lambda model, value: setattr(self.jobsModel, "stderr", value)
        self.stderr.getter = lambda model: self.jobsModel.stderr

        self.target_file = PathChooser(self, "", "install_job_target_file", show_files=True, must_be_set=False,
                                       must_exist=False)
        self.target_file.setter = lambda model, value: setattr(self.jobsModel, "target_file", value)
        self.target_file.getter = lambda model: self.jobsModel.target_file

        self.portable_exe = PathChooser(self, "", "install_job_portable_exe", show_files=True, must_be_set=False,
                                        must_exist=False)
        self.portable_exe.setter = lambda model, value: setattr(self.jobsModel, "portable_exe", value)
        self.portable_exe.getter = lambda model: self.jobsModel.portable_exe

        self.platform_exe = KeywordTable(self, "", "install_job_portable_exe", colHead1="Platform",
                                         colHead2="Executable")
        self.platform_exe.setter = lambda model, value: setattr(self.jobsModel, "platform_exe", value)
        self.platform_exe.getter = lambda model: self.jobsModel.platform_exe

        self.init_code = StringBox(self, "", "install_job_init_code")
        self.init_code.setter = lambda model, value: setattr(self.jobsModel, "init_code", value)
        self.init_code.getter = lambda model: self.jobsModel.init_code

        self.env = KeywordTable(self, "", "install_job_env", colHead1="Variable", colHead2="Value")
        self.env.setter = lambda model, value: setattr(self.jobsModel, "env", value)
        self.env.getter = lambda model: self.jobsModel.env

        layout.addRow("Stdin:", self.stdin)
        layout.addRow("Stdout:", self.stdout)
        layout.addRow("Stderr:", self.stderr)
        layout.addRow("Target file:", self.target_file)
        layout.addRow("Portable exe.:", self.portable_exe)
        layout.addRow("Platform exe.:", self.platform_exe)
        layout.addRow("Init code.:", self.init_code)
        layout.addRow("Env.:", self.env)

        self.setLayout(layout)


    def setJobModel(self, jobsModel):
        self.jobsModel = jobsModel

        self.stdin.fetchContent()
        self.stdout.fetchContent()
        self.stderr.fetchContent()
        self.target_file.fetchContent()
        self.portable_exe.fetchContent()
        self.platform_exe.fetchContent()
        self.init_code.fetchContent()
        self.env.fetchContent()
