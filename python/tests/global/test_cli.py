from tests import ErtTest
from res.test import ErtTestContext
import os

class EntryPointTest(ErtTest):
    def test_single_realization(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        exec_path = os.path.join(self.SOURCE_ROOT, "python/python/bin/ert_cli")
        os.execvp(exec_path, [exec_path, config_file, "--algorithm=Test-run"])

    def test_smoother_ensemble(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        exec_path = os.path.join(self.SOURCE_ROOT, "python/python/bin/ert_cli")
        os.execvp(exec_path, [exec_path, config_file, "--algorithm=Ensemble Smoother"])
