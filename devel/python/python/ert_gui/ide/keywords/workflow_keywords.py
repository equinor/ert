from ert_gui.ide.keywords.definitions import IntegerArgument, KeywordDefinition, ConfigurationLineDefinition, PathArgument, StringArgument, BoolArgument


class WorkflowKeywords(object):
    def __init__(self, ert_keywords):
        super(WorkflowKeywords, self).__init__()
        self.group = "Workflow Jobs"

        ert_keywords.addKeyword(self.addLoadWorkflowJob())
        ert_keywords.addKeyword(self.addWorkflowJobDirectory())
        ert_keywords.addKeyword(self.addLoadWorkflow())



    def addLoadWorkflowJob(self):
        load_workflow_job = ConfigurationLineDefinition(keyword=KeywordDefinition("LOAD_WORKFLOW_JOB"),
                                                        arguments=[StringArgument()],
                                                        documentation_link="workflow_jobs/load_workflow_job",
                                                        required=False,
                                                        group=self.group)
        return load_workflow_job



    def addWorkflowJobDirectory(self):
        workflow_job_directory = ConfigurationLineDefinition(keyword=KeywordDefinition("WORKFLOW_JOB_DIRECTORY"),
                                                        arguments=[PathArgument()],
                                                        documentation_link="workflow_jobs/workflow_job_directory",
                                                        required=False,
                                                        group=self.group)
        return workflow_job_directory


    def addLoadWorkflow(self):
        load_workflow = ConfigurationLineDefinition(keyword=KeywordDefinition("LOAD_WORKFLOW"),
                                                        arguments=[PathArgument()],
                                                        documentation_link="workflow_jobs/load_workflow",
                                                        required=False,
                                                        group=self.group)
        return load_workflow