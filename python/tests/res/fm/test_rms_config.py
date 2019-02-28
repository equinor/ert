#  Copyright (C) 2018  Equinor ASA, Norway.
#
#  The file 'test_ecl.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
import os
import stat
import unittest
import yaml
from ecl.util.test import TestAreaContext
from tests import ResTest

from res.fm.rms import RMSConfig


class RMSConfigTest(ResTest):

    def setUp(self):
        pass

    def test_load(self):
        os.environ["RMS_SITE_CONFIG"] = "file/does/not/exist"
        with self.assertRaises(IOError):
            conf = RMSConfig()

        os.environ["RMS_SITE_CONFIG"] = os.path.join(self.SOURCE_ROOT, "python/res/fm/rms/rms_config.yml")
        conf = RMSConfig()

        with self.assertRaises(OSError):
            exe = conf.executable


        with TestAreaContext("yaml"):
            with open("file.yml","w") as f:
                f.write("this:\n -should\n-be\ninvalid:yaml?")

            os.environ["RMS_SITE_CONFIG"] = "file.yml"
            with self.assertRaises(ValueError):
                conf = RMSConfig()

            os.mkdir("bin")
            with open("bin/rms", "w") as f:
                f.write("This is an RMS executable ...")
            os.chmod("bin/rms", stat.S_IEXEC)

            with open("file.yml", "w") as f:
                f.write("executable: bin/rms")

            conf = RMSConfig()
            self.assertEqual(conf.executable, "bin/rms")
            self.assertIsNone(conf.threads)

            with open("file.yml", "w") as f:
                f.write("executable: bin/rms\n")
                f.write("threads: 17")

            conf = RMSConfig()
            self.assertEqual(conf.threads, 17)

if __name__ == "__main__":
    unittest.main()
