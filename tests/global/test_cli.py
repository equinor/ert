from tests import ErtTest
from res.test import ErtTestContext
import os
import subprocess

class EntryPointTest(ErtTest):
    def test_single_realization(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        exec_path = os.path.join(self.SOURCE_ROOT, "python/python/bin/ert_cli")
        ok = subprocess.call([exec_path, config_file, "test_run", "default"])
        assert ok == 0
