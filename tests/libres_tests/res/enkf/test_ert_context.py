from distutils.errors import DistutilsFileError

from ...libres_utils import ResTest

from ert._c_wrappers.test import ErtTestContext


class ErtTestContextTest(ResTest):
    def setUp(self):
        self.config = self.createTestPath("local/snake_oil/snake_oil.ert")

    def test_raises(self):
        with self.assertRaises(DistutilsFileError):
            with ErtTestContext("Does/not/exist"):
                pass

    def test_workflow_ert_script_jobs(self):

        with ErtTestContext(self.config) as context:
            with self.assertRaises(IOError):
                context.installWorkflowJob("JOB_NAME", "DOES/NOT/EXIST")
