from tests import ErtTest
from res.test import ErtTestContext
import os
import subprocess

class EntryPointTest(ErtTest):
    def test_single_realization(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        exec_path = os.path.join(self.SOURCE_ROOT, "ert_gui/bin/ert_cli")
        ok = subprocess.call([exec_path, config_file, "test_run", "default"])
        assert ok == 0

    def test_ert_no_arguments(self):
        exec_path = os.path.join(self.SOURCE_ROOT, "bin/ert.in")
        proc = subprocess.Popen(exec_path, stderr=subprocess.PIPE)
        _, err = proc.communicate()

        assert("ert.in: error" in str(err))

