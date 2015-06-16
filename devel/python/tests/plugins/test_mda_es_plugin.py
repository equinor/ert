from ert.job_queue import ErtScript
from ert.test.extended_testcase import ExtendedTestCase
from tests.gui.ertshell.ert_shell_test_context import ErtShellTestContext


class MDAEnsembleSmootherPluginTest(ExtendedTestCase):

    def getWorkflowJob(self, ert, plugin_name):
        """ @rtype: WorkflowJob """
        plugin_name = plugin_name.strip()
        plugin_jobs = ert.getWorkflowList().getPluginJobs()
        plugin_job = next((job for job in plugin_jobs if job.name() == plugin_name), None)
        return plugin_job

    def getScript(self, ert, plugin_job):
        script_obj = ErtScript.loadScriptFromFile(plugin_job.getInternalScriptPath())
        script = script_obj(ert)
        return script

    def test_weights(self):
        test_config = self.createTestPath("local/custom_kw/mini_config")

        with ErtShellTestContext("python/workflow_jobs/mda_es", test_config) as shell:
            ert = shell.shellContext().ert()
            plugin_job = self.getWorkflowJob(ert, "MDA_ES")
            self.assertIsNotNone(plugin_job)
            script = self.getScript(ert, plugin_job)

            weights = script.parseWeights("iteration_weights/constant_4")
            self.assertAlmostEqualList([2, 2, 2, 2], weights)

            weights = script.parseWeights("iteration_weights/constant_2")
            self.assertAlmostEqualList([1.414213562373095, 1.414213562373095], weights)

            with self.assertRaises(ValueError):
                script.parseWeights("iteration_weights/error_in_weights")

            with self.assertRaises(ValueError):
                script.parseWeights("")

            weights = script.parseWeights("2, 2, 2, 2")
            self.assertAlmostEqualList([2, 2, 2, 2], weights)

            weights = script.parseWeights("1.414213562373095, 1.414213562373095")
            self.assertAlmostEqualList([1.414213562373095, 1.414213562373095], weights)

            with self.assertRaises(ValueError):
                script.parseWeights("2, error, 2, 2")


    def test_normalized_weights(self):
        test_config = self.createTestPath("local/custom_kw/mini_config")

        with ErtShellTestContext("python/workflow_jobs/mda_es", test_config) as shell:
            ert = shell.shellContext().ert()
            plugin_job = self.getWorkflowJob(ert, "MDA_ES")
            self.assertIsNotNone(plugin_job)
            script = self.getScript(ert, plugin_job)

            weights = script.normalizeWeights([1])
            self.assertAlmostEqualList([1.0], weights)

            weights = script.normalizeWeights([1, 1])
            self.assertAlmostEqualList([1.414214, 1.414214], weights)

            weights = script.normalizeWeights([1, 1, 1])
            self.assertAlmostEqualList([1.732051, 1.732051, 1.732051], weights)

            weights = script.normalizeWeights([8, 4, 2, 1])
            self.assertAlmostEqualList([9.219544457292887, 4.6097722286464435, 2.3048861143232218, 1.1524430571616109], weights)

            weights = script.normalizeWeights([9.219544457292887, 4.6097722286464435, 2.3048861143232218, 1.1524430571616109])
            self.assertAlmostEqualList([9.219544457292887, 4.6097722286464435, 2.3048861143232218, 1.1524430571616109], weights)