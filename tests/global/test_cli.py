from tests import ErtTest
from res.test import ErtTestContext
import os
import subprocess


class EntryPointTest(ErtTest):
    def test_single_realization(self):
        config_file = self.createTestPath('local/poly_example/poly.ert')
        exec_path = os.path.join(self.SOURCE_ROOT, "ert_gui/bin/ert")
        try:
            subprocess.check_call([exec_path, "cli", config_file, "--mode", "test_run",
                                   "--target-case", "default"])
        except subprocess.CalledProcessError as e:
            self.fail("%s: %s" % (e.__class__.__name__, e))
