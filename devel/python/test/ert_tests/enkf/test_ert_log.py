import os
from ert.enkf import ErtLog
from ert.test import ExtendedTestCase


class ErtLogTest(ExtendedTestCase):

    def test_write_log(self):
        test_log_filename = "/tmp/test_log"
        ErtLog.cnamespace.init(1, test_log_filename, "unittester", True)
        message = "This is fun"
        ErtLog.log(1, message)

        self.assertTrue(os.path.isfile(test_log_filename))

        with open(test_log_filename, "r") as f:
            text = f.readlines()
            self.assertTrue(len(text) > 0)
            self.assertTrue(message in text[-1])