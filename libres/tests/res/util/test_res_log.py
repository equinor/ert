import os

from ecl.util.test import TestAreaContext
from utils import ResTest

from res.util import Log, ResLog
from res.util.enums import MessageLevelEnum


class ResLogTest(ResTest):
    def test_log(self):
        with TestAreaContext("python/res_log/log") as work_area:
            test_log_filename = "test_log"
            ResLog.init(1, test_log_filename, True)
            message = "This is fun"
            ResLog.log(1, message)

            self.assertTrue(os.path.isfile(test_log_filename))

            with open(test_log_filename, "r") as f:
                text = f.readlines()
                self.assertTrue(len(text) > 0)
                self.assertTrue(message in text[-1])

    def test_get_filename(self):
        with TestAreaContext("python/res_log/log") as work_area:
            test_log_filename = "log_test_file.txt"
            ResLog.init(1, test_log_filename, True)
            message = "This is fun"
            ResLog.log(1, message)

            self.assertEqual(ResLog.getFilename(), test_log_filename)
