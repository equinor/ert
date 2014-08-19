from ert.job_queue import WorkflowJoblist, WorkflowJob
from ert.test import ExtendedTestCase, TestAreaContext


class WorkflowTest(ExtendedTestCase):

    def test_workflow_joblist_creation(self):
        joblist = WorkflowJoblist()

        job = WorkflowJob("JOB1")

        joblist.addJob(job)

        self.assertTrue(job in joblist)
        self.assertTrue("JOB1" in joblist)

        job_ref = joblist["JOB1"]

        self.assertEqual(job.name(), job_ref.name())


    def createWorkflowJobs(self):
        with open("workflow_job_1", "w") as f:
            f.write("INTERNAL True\n")
            f.write("FUNCTION enkf_main_select_case_JOB\n")
            f.write("MIN_ARG 1\n")
            f.write("MAX_ARG 1\n")
            f.write("ARG_TYPE 0 STRING\n")

        with open("workflow_job_2", "w") as f:
            f.write("INTERNAL True\n")
            f.write("SCRIPT /path/to/script\n")
            f.write("MIN_ARG 1\n")
            f.write("MAX_ARG 1\n")
            f.write("ARG_TYPE 0 STRING\n")


    def test_workflow_joblist_with_files(self):
        with TestAreaContext("python/job_queue/workflow_joblist") as work_area:
            self.createWorkflowJobs()

            joblist = WorkflowJoblist()

            joblist.addJobFromFile("FILE_JOB", "workflow_job_1")
            joblist.addJobFromFile("SCRIPT_JOB", "workflow_job_2")

            self.assertTrue("FILE_JOB" in joblist)
            self.assertTrue("SCRIPT_JOB" in joblist)

            job = joblist["SCRIPT_JOB"]
            self.assertTrue(job.isInternalScript())
