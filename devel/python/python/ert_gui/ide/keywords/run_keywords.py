from ert_gui.ide.keywords.definitions import IntegerArgument, KeywordDefinition, ConfigurationLineDefinition, PathArgument, StringArgument


class RunKeywords(object):
    def __init__(self, ert_keywords):
        super(RunKeywords, self).__init__()
        self.group = "Run"

        ert_keywords.addKeyword(self.addDeleteRunpath())
        ert_keywords.addKeyword(self.addKeepRunpath())
        ert_keywords.addKeyword(self.addInstallJob())
        ert_keywords.addKeyword(self.addRunpath())
        ert_keywords.addKeyword(self.addRerunPath())
        ert_keywords.addKeyword(self.addRunpathFile())
        ert_keywords.addKeyword(self.addForwardModel())
        ert_keywords.addKeyword(self.addJobScript())
        ert_keywords.addKeyword(self.addRunTemplate())







    def addInstallJob(self):
        install_job = ConfigurationLineDefinition(keyword=KeywordDefinition("INSTALL_JOB"),
                                                  arguments=[StringArgument(),PathArgument()],
                                                  documentation_link="run/install_job",
                                                  required=False,
                                                  group=self.group)
        return install_job



    def addDeleteRunpath(self):
        delete_runpath = ConfigurationLineDefinition(keyword=KeywordDefinition("DELETE_RUNPATH"),
                                                     arguments=[StringArgument()],
                                                     documentation_link="run/delete_runpath",
                                                     required=False,
                                                     group=self.group)
        return delete_runpath


    def addKeepRunpath(self):
        keep_runpath = ConfigurationLineDefinition(keyword=KeywordDefinition("KEEP_RUNPATH"),
                                                   arguments=[StringArgument()],
                                                   documentation_link="run/keep_runpath",
                                                   required=False,
                                                   group=self.group)
        return keep_runpath





    def addRunpath(self):
        runpath = ConfigurationLineDefinition(keyword=KeywordDefinition("RUNPATH"),
                                                  arguments=[PathArgument()],
                                                  documentation_link="run/runpath",
                                                  required=False,
                                                  group=self.group)
        return runpath


    def addRerunPath(self):
        rerun_path = ConfigurationLineDefinition(keyword=KeywordDefinition("RERUN_PATH"),
                                                  arguments=[PathArgument()],
                                                  documentation_link="run/rerun_path",
                                                  required=False,
                                                  group=self.group)
        return rerun_path



    def addRunpathFile(self):
        runpath_file = ConfigurationLineDefinition(keyword=KeywordDefinition("RUNPATH_FILE"),
                                                  arguments=[PathArgument()],
                                                  documentation_link="run/runpath_file",
                                                  required=False,
                                                  group=self.group)
        return runpath_file


    def addForwardModel(self):
        forward_model = ConfigurationLineDefinition(keyword=KeywordDefinition("FORWARD_MODEL"),
                                                    arguments=[StringArgument(),
                                                               StringArgument(rest_of_line=True, allow_space=True)],
                                                    documentation_link="run/forward_model",
                                                    required=False,
                                                    group=self.group)
        return forward_model

    def addJobScript(self):
        job_script = ConfigurationLineDefinition(keyword=KeywordDefinition("JOB_SCRIPT"),
                                                 arguments=[PathArgument()],
                                                 documentation_link="run/job_script",
                                                 required=False,
                                                 group=self.group)
        return job_script

    def addRunTemplate(self):
        run_template = ConfigurationLineDefinition(keyword=KeywordDefinition("RUN_TEMPLATE"),
                                                 arguments=[PathArgument(),StringArgument()],
                                                 documentation_link="run/run_template",
                                                 required=False,
                                                 group=self.group)
        return run_template