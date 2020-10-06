import os
from res.util import ResLog, Log
from ecl.util.test import TestAreaContext
from tests import ResTest
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

    def test_getFilename(self):
        with TestAreaContext("python/res_log/log") as work_area:
            test_log_filename = "log_test_file.txt"
            ResLog.init(1, test_log_filename, True)
            message = "This is fun"
            ResLog.log(1, message)

            self.assertEqual(ResLog.getFilename(), test_log_filename)

    def test_log(self):
        with TestAreaContext("python/log"):
            logh = Log("logfile", MessageLevelEnum.LOG_DEBUG)

            os.mkdir("read_only")
            os.chmod("read_only", 0o500)
            with self.assertRaises(IOError):
                logh = Log("read_only/logfile.txt", MessageLevelEnum.LOG_DEBUG)

    def test_init_perm_denied(self):
        with TestAreaContext("python/res_log"):
            os.mkdir("read_only")
            os.chmod("read_only", 0o500)
            with self.assertRaises(IOError):
                ResLog.init(1, "read_only/logfile.txt", True)
