from ert_gui.ide.keywords.definitions import IntegerArgument, KeywordDefinition, ConfigurationLineDefinition, PathArgument, StringArgument


class RunKeywords(object):
    def __init__(self, ert_keywords):
        super(RunKeywords, self).__init__()
        self.group = "Run"

        ert_keywords.addKeyword(self.addDeleteRunpath())
        ert_keywords.addKeyword(self.addKeepRunpath())
        ert_keywords.addKeyword(self.addInstallJob())
        ert_keywords.addKeyword(self.addRunpath())
        ert_keywords.addKeyword(self.addRunpathFile())
        ert_keywords.addKeyword(self.addForwardModel())
        ert_keywords.addKeyword(self.addJobScript())






    def addInstallJob(self):
        install_job = ConfigurationLineDefinition(keyword=KeywordDefinition("INSTALL_JOB"),
                                                  arguments=[StringArgument(),PathArgument()],
                                                  documentation_link="ensemble/install_job",
                                                  required=False,
                                                  group=self.group)
        return install_job



    def addDeleteRunpath(self):
        delete_runpath = ConfigurationLineDefinition(keyword=KeywordDefinition("DELETE_RUNPATH"),
                                                     arguments=[StringArgument()],
                                                     documentation_link="ensemble/delete_runpath",
                                                     required=False,
                                                     group=self.group)
        return delete_runpath


    def addKeepRunpath(self):
        keep_runpath = ConfigurationLineDefinition(keyword=KeywordDefinition("KEEP_RUNPATH"),
                                                   arguments=[StringArgument()],
                                                   documentation_link="ensemble/keep_runpath",
                                                   required=False,
                                                   group=self.group)
        return keep_runpath





    def addRunpath(self):
        runpath = ConfigurationLineDefinition(keyword=KeywordDefinition("RUNPATH"),
                                                  arguments=[PathArgument()],
                                                  documentation_link="ensemble/runpath",
                                                  required=False,
                                                  group=self.group)
        return runpath


    def addRunpathFile(self):
        runpath_file = ConfigurationLineDefinition(keyword=KeywordDefinition("RUNPATH_FILE"),
                                                  arguments=[PathArgument()],
                                                  documentation_link="ensemble/runpath_file",
                                                  required=False,
                                                  group=self.group)
        return runpath_file


    def addForwardModel(self):
        forward_model = ConfigurationLineDefinition(keyword=KeywordDefinition("FORWARD_MODEL"),
                                                    arguments=[StringArgument(),
                                                               StringArgument(rest_of_line=True, allow_space=True)],
                                                    documentation_link="ensemble/forward_model",
                                                    required=False,
                                                    group=self.group)
        return forward_model

    def addJobScript(self):
        job_script = ConfigurationLineDefinition(keyword=KeywordDefinition("JOB_SCRIPT"),
                                                 arguments=[PathArgument()],
                                                 documentation_link="ensemble/job_script",
                                                 required=False,
                                                 group=self.group)
        return job_script