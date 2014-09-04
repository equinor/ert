from ert.test import ErtTestContext, ExtendedTestCase


class ErtTestContextTest(ExtendedTestCase):
    def setUp(self):
        self.config = self.createTestPath("Statoil/config/with_data/config")

    def test_raises(self):
        with self.assertRaises(IOError):
            testContext = ErtTestContext("ExistTest" , "Does/not/exist")


    def initFromCaseTest(self, context):
        ert = context.getErt()

        init_case_job = self.createSharePath("workflows/jobs/internal/config/INIT_CASE_FROM_EXISTING")
        self.assertTrue(context.installWorkflowJob("INIT_CASE_JOB", init_case_job))
        self.assertTrue(context.runWorkflowJob("INIT_CASE_JOB", "default", "new_not_current_case"))

        default_fs = ert.getEnkfFsManager().getFileSystem("default")
        new_fs = ert.getEnkfFsManager().getFileSystem("new_not_current_case")

        self.assertIsNotNone(default_fs)
        self.assertIsNotNone(new_fs)

        self.assertTrue(len(default_fs.getStateMap()) > 0)
        self.assertEqual(len(default_fs.getStateMap()), len(new_fs.getStateMap()))


    def createCaseTest(self, context):
        create_case_job = self.createSharePath("workflows/jobs/internal/config/CREATE_CASE")
        self.assertTrue(context.installWorkflowJob("CREATE_CASE_JOB", create_case_job))
        self.assertTrue(context.runWorkflowJob("CREATE_CASE_JOB", "newly_created_case"))
        self.assertDirectoryExists("storage/newly_created_case")

    def selectCaseTest(self, context):
        ert = context.getErt()
        select_case_job = self.createSharePath("workflows/jobs/internal/config/SELECT_CASE")

        default_fs = ert.getEnkfFsManager().getCurrentFileSystem()

        custom_fs = ert.getEnkfFsManager().getFileSystem("CustomCase")

        self.assertEqual(ert.getEnkfFsManager().getCurrentFileSystem(), default_fs)

        self.assertTrue(context.installWorkflowJob("SELECT_CASE_JOB", select_case_job))
        self.assertTrue(context.runWorkflowJob("SELECT_CASE_JOB", "CustomCase"))

        self.assertEqual(ert.getEnkfFsManager().getCurrentFileSystem(), custom_fs)


    def loadResultsTest(self, context):
        load_results_job = self.createSharePath("workflows/jobs/internal/config/LOAD_RESULTS")
        self.assertTrue(context.installWorkflowJob("LOAD_RESULTS_JOB", load_results_job))
        self.assertTrue(context.runWorkflowJob("LOAD_RESULTS_JOB", 0, 1))


    def rankRealizationsOnObservationsTest(self, context):
        rank_job = self.createSharePath("workflows/jobs/internal/config/OBSERVATION_RANKING")

        self.assertTrue(context.installWorkflowJob("OBS_RANK_JOB", rank_job))

        self.assertTrue(context.runWorkflowJob("OBS_RANK_JOB", "NameOfObsRanking1", "|", "WOPR:*"))
        self.assertTrue(context.runWorkflowJob("OBS_RANK_JOB", "NameOfObsRanking2", "1-5", "55", "|", "WWCT:*", "WOPR:*"))
        self.assertTrue(context.runWorkflowJob("OBS_RANK_JOB", "NameOfObsRanking3", "5", "55", "|"))
        self.assertTrue(context.runWorkflowJob("OBS_RANK_JOB", "NameOfObsRanking4", "1,3,5-10", "55"))
        self.assertTrue(context.runWorkflowJob("OBS_RANK_JOB", "NameOfObsRanking5"))
        self.assertTrue(context.runWorkflowJob("OBS_RANK_JOB", "NameOfObsRanking6", "|", "UnrecognizableObservation"))


    def test_workflow_jobs(self):

        with ErtTestContext("python/enkf/ert_test_context_workflow_job", self.config) as context:
            self.createCaseTest(context)
            self.selectCaseTest(context)
            self.initFromCaseTest(context)
            self.loadResultsTest(context)
            self.rankRealizationsOnObservationsTest(context)




