import pkg_resources
import pytest
from libres_utils import ResTest

from res.test import ErtTestContext


@pytest.mark.equinor_test
class ErtTestContextTest(ResTest):
    def setUp(self):
        self.config = self.createTestPath("Equinor/config/with_data/config")

    def test_raises(self):
        with self.assertRaises(IOError):
            testContext = ErtTestContext("ExistTest", "Does/not/exist")

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

    def loadResultsTest(self, context):
        resource_file = pkg_resources.resource_filename(
            "ert_shared", "share/ert/workflows/jobs/internal/config/LOAD_RESULTS"
        )

        context.installWorkflowJob("LOAD_RESULTS_JOB", resource_file)
        self.assertTrue(context.runWorkflowJob("LOAD_RESULTS_JOB", 0, 1))

    def rankRealizationsOnObservationsTest(self, context):
        rank_job = pkg_resources.resource_filename(
            "ert_shared", "share/ert/workflows/jobs/internal/config/OBSERVATION_RANKING"
        )
        context.installWorkflowJob("OBS_RANK_JOB", rank_job)

        self.assertTrue(
            context.runWorkflowJob("OBS_RANK_JOB", "NameOfObsRanking1", "|", "WOPR:*")
        )
        self.assertTrue(
            context.runWorkflowJob(
                "OBS_RANK_JOB",
                "NameOfObsRanking2",
                "1-5",
                "55",
                "|",
                "WWCT:*",
                "WOPR:*",
            )
        )
        self.assertTrue(
            context.runWorkflowJob("OBS_RANK_JOB", "NameOfObsRanking3", "5", "55", "|")
        )
        self.assertTrue(
            context.runWorkflowJob(
                "OBS_RANK_JOB", "NameOfObsRanking4", "1,3,5-10", "55"
            )
        )
        self.assertTrue(context.runWorkflowJob("OBS_RANK_JOB", "NameOfObsRanking5"))
        self.assertTrue(
            context.runWorkflowJob(
                "OBS_RANK_JOB", "NameOfObsRanking6", "|", "UnrecognizableObservation"
            )
        )

    def test_workflow_function_jobs(self):

        with ErtTestContext(
            "python/enkf/ert_test_context_workflow_function_job", self.config
        ) as context:
            internal_config = "share/ert/workflows/jobs/internal-tui/config"
            self.createCaseTest(context, root_path=internal_config)
            self.selectCaseTest(context, root_path=internal_config)

            # Due to EnKFFs caching and unmonitored C functions this will fail
            # self.initFromCaseTest(context, root_path=internal_config)

            self.loadResultsTest(context)
            self.rankRealizationsOnObservationsTest(context)

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
