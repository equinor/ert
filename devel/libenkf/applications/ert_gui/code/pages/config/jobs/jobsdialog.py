class InstallJobsPanel(HelpedWidget):
    """
    A panel for creating custom jobs.
    Any created jobs are automatically stored in: guijobs/job_name
    """
    def __init__(self, parent=None):
        HelpedWidget.__init__(self, parent, "", "install_jobs")

        self.searchableList = SearchableList(parent, listHeight=200)
        self.addWidget(self.searchableList)

        self.connect(self.searchableList, QtCore.SIGNAL('currentItemChanged(QListWidgetItem, QListWidgetItem)'),
                     self.changeParameter)
        self.connect(self.searchableList, QtCore.SIGNAL('addItem(list)'), self.addItem)
        self.connect(self.searchableList, QtCore.SIGNAL('removeItem(list)'), self.removeItem)

        self.jobPanel = JobConfigPanel(parent)

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
    def __init__(self, name, path=None):
        self.name = name

        if path is None:
            self.path = "private_jobs/" + name
        else:
            self.path = path

        self.stdin = ""
        self.stdout = ""
        self.stderr = ""
        self.target_file = ""
        self.portable_exe = ""
        self.platform_exe = []
        self.init_code = []
        self.env = []
        self.arglist = ""
        self.lsf_resources = ""

    def set(self, attr, value):
        setattr(self, attr, value)


class JobConfigPanel(ConfigPanel):
    def __init__(self, parent=None):
        ConfigPanel.__init__(self, parent)

        self.jobsModel = JobsModel("")

        #self.setFrameShape(QtGui.QFrame.StyledPanel)
        #self.setFrameShadow(QtGui.QFrame.Plain)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        self.stdin = PathChooser(self, "", "install_job_stdin", show_files=True, must_be_set=False)
        self.stdin.setter = lambda model, value: self.set("stdin", value)
        self.stdin.getter = lambda model: self.jobsModel.stdin

        self.stdout = PathChooser(self, "", "install_job_stdout", show_files=True, must_be_set=False, must_exist=False)
        self.stdout.setter = lambda model, value: self.set("stdout", value)
        self.stdout.getter = lambda model: self.jobsModel.stdout

        self.stderr = PathChooser(self, "", "install_job_stderr", show_files=True, must_be_set=False, must_exist=False)
        self.stderr.setter = lambda model, value: self.set("stderr", value)
        self.stderr.getter = lambda model: self.jobsModel.stderr

        self.target_file = PathChooser(self, "", "install_job_target_file", show_files=True, must_be_set=False,
                                       must_exist=False)
        self.target_file.setter = lambda model, value: self.set("target_file", value)
        self.target_file.getter = lambda model: self.jobsModel.target_file

        self.portable_exe = PathChooser(self, "", "install_job_portable_exe", show_files=True, must_be_set=False,
                                        must_exist=False)
        self.portable_exe.setter = lambda model, value: self.set("portable_exe", value)
        self.portable_exe.getter = lambda model: self.jobsModel.portable_exe

        self.platform_exe = KeywordTable(self, "", "install_job_platform_exe", colHead1="Platform",
                                         colHead2="Executable")
        self.platform_exe.setter = lambda model, value: self.set("platform_exe", value)
        self.platform_exe.getter = lambda model: self.jobsModel.platform_exe

        self.init_code = KeywordList(self, "", "install_job_init_code")
        self.init_code.setter = lambda model, value: self.set("init_code", value)
        self.init_code.getter = lambda model: self.jobsModel.init_code

        self.env = KeywordTable(self, "", "install_job_env", colHead1="Variable", colHead2="Value")
        self.env.setter = lambda model, value: self.set("env", value)
        self.env.getter = lambda model: self.jobsModel.env

        self.arglist = StringBox(self, "", "install_job_arglist")
        self.arglist.setter = lambda model, value: self.set("arglist", value)
        self.arglist.getter = lambda model: self.jobsModel.arglist

        self.lsf_resources = StringBox(self, "", "install_job_lsf_resources")
        self.lsf_resources.setter = lambda model, value: self.set("lsf_resources", value)
        self.lsf_resources.getter = lambda model: self.jobsModel.lsf_resources


        self.startPage("Standard")
        self.add("Stdin:", self.stdin)
        self.add("Stdout:", self.stdout)
        self.add("Stderr:", self.stderr)
        self.add("Target file:", self.target_file)
        self.add("Portable exe.:", self.portable_exe)
        self.add("Env.:", self.env)
        self.endPage()

        self.startPage("Advanced")
        self.add("Platform exe.:", self.platform_exe)
        self.add("Init code.:", self.init_code)
        self.add("Arglist.:", self.arglist)
        self.add("LSF resources:", self.lsf_resources)
        self.endPage()

    def add(self, label, widget):
        self.addRow(widget, label)


    def set(self, attr, value):
        self.jobsModel.set(attr, value)

        f = open(self.jobsModel.path, 'w')

        if not self.jobsModel.stdin == "":
            f.write("STDIN " + self.jobsModel.stdin + "\n")

        if not self.jobsModel.stdout == "":
            f.write("STDOUT " + self.jobsModel.stdout + "\n")

        if not self.jobsModel.stderr == "":
            f.write("STDERR " + self.jobsModel.stderr + "\n")

        if not self.jobsModel.target_file == "":
            f.write("TARGET_FILE " + self.jobsModel.target_file + "\n")

        if not self.jobsModel.portable_exe == "":
            f.write("PORTABLE_EXE " + self.jobsModel.portable_exe + "\n")

        if not self.jobsModel.arglist == "":
            f.write("ARGLIST " + self.jobsModel.arglist + "\n")

        if not self.jobsModel.lsf_resources == "":
            f.write("LSF_RESOURCES " + self.jobsModel.lsf_resources + "\n")

        if not self.jobsModel.platform_exe is None:
            for k, v in self.jobsModel.platform_exe:
                f.write("PLATFORM_EXE " + k + " " + v + "\n")

        if not self.jobsModel.init_code is None:
            for v in self.jobsModel.init_code:
                f.write("INIT_CODE " + v + "\n")

        if not self.jobsModel.env is None:
            for k, v in self.jobsModel.env:
                f.write("ENV " + k + " " + v + "\n")

        f.close()

    def setJobModel(self, jobsModel):
        self.jobsModel = jobsModel

        path = jobsModel.path

        if os.path.exists(path) and os.path.isfile(path):
            f = open(path, 'r')
            data = f.read()
            self.matchAndApply("STDIN +(.*)", data, "stdin")
            self.matchAndApply("STDOUT +(.*)", data, "stdout")
            self.matchAndApply("STDERR +(.*)", data, "stderr")
            self.matchAndApply("TARGET_FILE +(.*)", data, "target_file")
            self.findAllAndApply("INIT_CODE +(.*)", data, "init_code", groups=1)
            self.matchAndApply("PORTABLE_EXE +(.*)", data, "portable_exe")
            self.matchAndApply("ARGLIST +(.*)", data, "arglist")
            self.matchAndApply("LSF_RESOURCES +(.*)", data, "lsf_resources")

            self.findAllAndApply("PLATFORM_EXE +([^ ]*) +(.*)", data, "platform_exe")
            self.findAllAndApply("ENV +([^ ]*) +(.*)", data, "env")

            f.close()

        self.stdin.fetchContent()
        self.stdout.fetchContent()
        self.stderr.fetchContent()
        self.target_file.fetchContent()
        self.portable_exe.fetchContent()
        self.platform_exe.fetchContent()
        self.init_code.fetchContent()
        self.env.fetchContent()
        self.arglist.fetchContent()
        self.lsf_resources.fetchContent()

    def matchAndApply(self, regex, data, attr):
        matcher = re.search(regex, data)
        if matcher:
            self.jobsModel.set(attr, matcher.group(1))

    def findAllAndApply(self, regex, data, attr, groups=2):
        matcher = re.findall(regex, data)
        if matcher:
            value = []
            for m in matcher:
                if groups == 2:
                    value.append([m[0], m[1]])
                else:
                    value.append(m)

            self.jobsModel.set(attr, value)
