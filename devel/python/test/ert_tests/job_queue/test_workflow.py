from ert.job_queue import Workflow, WorkflowJoblist
from ert.test import ExtendedTestCase, TestAreaContext


class WorkflowTest(ExtendedTestCase):


    def createWorkflowFiles(self):
        with open("workflow", "w") as f:
            f.write("PRINT Step1\n")
            f.write("PRINT Step2:AnotherString\n")


        with open("print_job", "w") as f:
            f.write("INTERNAL True\n")
            f.write("SCRIPT print\n")
            f.write("MIN_ARG 1\n")
            f.write("MAX_ARG 1\n")
            f.write("ARG_TYPE 0 STRING\n")


    def test_workflow(self):
        with TestAreaContext("python/job_queue/workflow") as work_area:
            self.createWorkflowFiles()

            joblist = WorkflowJoblist()
            joblist.addJobFromFile("PRINT", "print_job")

            self.assertTrue("PRINT" in joblist)


            workflow = Workflow("workflow", joblist)

            self.assertEqual(len(workflow), 2)

            job, args = workflow[0]
            self.assertEqual(job, joblist["PRINT"])
            self.assertEqual(args[0], "Step1")

            job, args = workflow[1]
            self.assertEqual(job, joblist["PRINT"])
            self.assertEqual(args[0], "Step2:AnotherString")
