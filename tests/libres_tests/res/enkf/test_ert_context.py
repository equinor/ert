import pkg_resources
from distutils.errors import DistutilsFileError

from libres_utils import ResTest

from res.test import ErtTestContext


class ErtTestContextTest(ResTest):
    def setUp(self):
        self.config = self.createTestPath("local/snake_oil/snake_oil.ert")

    def test_raises(self):
        with self.assertRaises(DistutilsFileError):
            with ErtTestContext("ExistTest", "Does/not/exist"):
                pass

    def initFromCaseTest(self, context, root_path):
        ert = context.getErt()
        resource_file = pkg_resources.resource_filename(
            "ert_shared", root_path + "/INIT_CASE_FROM_EXISTING"
        )

        context.installWorkflowJob("INIT_CASE_JOB", resource_file)
        self.assertTrue(
            context.runWorkflowJob("INIT_CASE_JOB", "default", "new_not_current_case")
        )

        default_fs = ert.getEnkfFsManager().getFileSystem("default_0")
        new_fs = ert.getEnkfFsManager().getFileSystem("new_not_current_case")

        self.assertIsNotNone(default_fs)
        self.assertIsNotNone(new_fs)

        self.assertTrue(len(default_fs.getStateMap()) > 0)
        self.assertEqual(len(default_fs.getStateMap()), len(new_fs.getStateMap()))

    def createCaseTest(self, context, root_path):
        resource_file = pkg_resources.resource_filename(
            "ert_shared", root_path + "/CREATE_CASE"
        )
        context.installWorkflowJob("CREATE_CASE_JOB", resource_file)
        self.assertFalse(
            context.getErt().getEnkfFsManager().caseExists("newly_created_case")
        )
        self.assertTrue(context.runWorkflowJob("CREATE_CASE_JOB", "newly_created_case"))
        self.assertTrue(
            context.getErt().getEnkfFsManager().caseExists("newly_created_case")
        )

    def selectCaseTest(self, context, root_path):
        ert = context.getErt()
        resource_file = pkg_resources.resource_filename(
            "ert_shared", root_path + "/SELECT_CASE"
        )

        default_fs = ert.getEnkfFsManager().getCurrentFileSystem()

        custom_fs = ert.getEnkfFsManager().getFileSystem("CustomCase")

        self.assertEqual(ert.getEnkfFsManager().getCurrentFileSystem(), default_fs)

        context.installWorkflowJob("SELECT_CASE_JOB", resource_file)
        self.assertTrue(context.runWorkflowJob("SELECT_CASE_JOB", "CustomCase"))

        self.assertEqual(ert.getEnkfFsManager().getCurrentFileSystem(), custom_fs)

    def test_workflow_function_jobs(self):

        with ErtTestContext(
            "python/enkf/ert_test_context_workflow_function_job", self.config
        ) as context:
            internal_config = "share/ert/workflows/jobs/internal-tui/config"
            self.createCaseTest(context, root_path=internal_config)
            self.selectCaseTest(context, root_path=internal_config)

    def test_workflow_ert_script_jobs(self):

        with ErtTestContext(
            "python/enkf/ert_test_context_workflow_ert_script_job", self.config
        ) as context:
            with self.assertRaises(IOError):
                context.installWorkflowJob("JOB_NAME", "DOES/NOT/EXIST")

            ert_scripts_config = "share/ert/workflows/jobs/internal-gui/config"
            self.createCaseTest(context, root_path=ert_scripts_config)
            self.selectCaseTest(context, root_path=ert_scripts_config)
            self.initFromCaseTest(context, root_path=ert_scripts_config)
