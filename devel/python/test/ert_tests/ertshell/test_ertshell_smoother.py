from ert.enkf.export import GenKwCollector
from ert.test.extended_testcase import ExtendedTestCase
from ert_tests.ertshell.ert_shell_test_context import ErtShellTestContext


class ErtShellSmootherTest(ExtendedTestCase):

    def test_smoother(self):
        test_config = self.createTestPath("local/custom_kw/mini_config")

        with ErtShellTestContext("python/ertshell/smoother", test_config) as shell:
            shell.invokeCommand("case select test_run")

            self.assertTrue(shell.invokeCommand("smoother update test_run_update"))

            shell.invokeCommand("case select test_run_update")

            ert = shell.shellContext().ert()
            data = GenKwCollector.loadAllGenKwData(ert, "test_run", keys=["PERLIN_PARAM:SCALE"])
            update_data = GenKwCollector.loadAllGenKwData(ert, "test_run_update", keys=["PERLIN_PARAM:SCALE"])

            self.assertTrue(data["PERLIN_PARAM:SCALE"].std() > update_data["PERLIN_PARAM:SCALE"].std())
