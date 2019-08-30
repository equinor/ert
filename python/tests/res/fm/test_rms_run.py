#  Copyright (C) 2018  Equinor ASA, Norway.
#
#  The file 'test_rms.py' is part of ERT - Ensemble based Reservoir Tool.
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
import subprocess
import unittest
import yaml
import shutil
import json
from ecl.util.test import TestAreaContext
from tests import ResTest

from res.fm.rms import RMSRun
import res.fm.rms

class RMSRunTest(ResTest):

    def setUp(self):
        pass

    def test_create(self):
        os.environ["RMS_SITE_CONFIG"] =os.path.join(self.SOURCE_ROOT, "python/res/fm/rms/rms_config.yml")
        with self.assertRaises(OSError):
            r = RMSRun(0, "/project/does/not/exist", "workflow")

        with TestAreaContext("test_create"):
            os.mkdir("rms")
            r = RMSRun(0, "rms", "workflow")


    def test_run_class(self):
        with TestAreaContext("test_run"):
            with open("rms_config.yml", "w") as f:
                f.write("executable:  {}/bin/rms".format(os.getcwd()))


            os.mkdir("run_path")
            os.mkdir("bin")
            os.mkdir("project")
            shutil.copy(os.path.join(self.SOURCE_ROOT, "python/tests/res/fm/rms"), "bin")
            os.environ["RMS_SITE_CONFIG"] = "rms_config.yml"

            action = {"exit_status" : 0}
            with open("run_path/action.json", "w") as f:
                f.write( json.dumps(action) )

            r = RMSRun(0, "project", "workflow", run_path="run_path")
            r.run()

            # -----------------------------------------------------------------

            action = {"exit_status" : 1}
            with open("run_path/action.json", "w") as f:
                f.write( json.dumps(action) )

            r = RMSRun(0, "project", "workflow", run_path="run_path")
            with self.assertRaises(Exception):
                r.run()

            # -----------------------------------------------------------------

            action = {"exit_status" : 0}
            with open("run_path/action.json", "w") as f:
                f.write( json.dumps(action) )

            r = RMSRun(0, "project", "workflow", run_path="run_path", target_file="some_file")
            with self.assertRaises(Exception):
                r.run()

            # -----------------------------------------------------------------

            action = {"exit_status" : 0,
                      "target_file" : os.path.join(os.getcwd(), "some_file")}
            with open("run_path/action.json", "w") as f:
                f.write( json.dumps(action) )

            r = RMSRun(0, "project", "workflow", run_path="run_path", target_file="some_file")
            r.run()



    def test_run(self):
        with TestAreaContext("test_run"):
            with open("rms_config.yml", "w") as f:
                f.write("executable:  {}/bin/rms".format(os.getcwd()))


            os.mkdir("run_path")
            os.mkdir("bin")
            os.mkdir("project")
            shutil.copy(os.path.join(self.SOURCE_ROOT, "python/tests/res/fm/rms"), "bin")
            os.environ["RMS_SITE_CONFIG"] = "rms_config.yml"

            action = {"exit_status" : 0}
            with open("run_path/action.json", "w") as f:
                f.write( json.dumps(action) )

            res.fm.rms.run(0, "project", "workflow", run_path="run_path")

            # -----------------------------------------------------------------

            action = {"exit_status" : 1}
            with open("run_path/action.json", "w") as f:
                f.write( json.dumps(action) )

            with self.assertRaises(Exception):
                res.fm.rms.run(0, "project", "workflow", run_path="run_path")

            # -----------------------------------------------------------------

            action = {"exit_status" : 0}
            with open("run_path/action.json", "w") as f:
                f.write( json.dumps(action) )

            with self.assertRaises(Exception):
                res.fm.rms.run(0, "project", "workflow", run_path="run_path", target_file="some_file")

            # -----------------------------------------------------------------

            action = {"exit_status" : 0,
                      "target_file" : os.path.join(os.getcwd(), "some_file")}

            with open("run_path/action.json", "w") as f:
                f.write( json.dumps(action) )
            res.fm.rms.run(0, "project", "workflow", run_path="run_path", target_file="some_file")

    def test_rms_load_env(self):
        test_bed = [
            ('    ', False),
            ('', False),
            (None, False),
            ('SOME_VAL', True),
        ]
        for val, carry_over in test_bed:
            with TestAreaContext('test_drop_path'):
                # Setup RMS project
                with open('rms_config.yml', 'w') as f:
                    json.dump({
                            'executable': os.path.realpath('bin/rms'),
                        }, f)

                with open('rms_exec_env.json', 'w') as f:
                    json.dump({
                        'RMS_TEST_VAR': val,
                    }, f)

                os.mkdir("run_path")
                os.mkdir("bin")
                os.mkdir("project")
                shutil.copy(os.path.join(self.SOURCE_ROOT, "python/tests/res/fm/rms"), "bin")
                os.environ["RMS_SITE_CONFIG"] = "rms_config.yml"

                action = {"exit_status" : 0}
                with open("run_path/action.json", "w") as f:
                    f.write( json.dumps(action) )

                rms_exec = os.path.join(self.SOURCE_ROOT, 'share/ert/forward-models/res/script/rms')
                subprocess.check_call([
                    rms_exec,
                    '--run-path',
                    'run_path',
                    '0',
                    '--version',
                    '10.4',
                    'project',
                    '--import-path',
                    './',
                    '--export-path',
                    './',
                    'workflow',
                ])

                with open('run_path/env.json') as f:
                    env = json.load(f)

                if carry_over:
                    self.assertIn('RMS_TEST_VAR', env)
                else:
                    self.assertNotIn('RMS_TEST_VAR', env)

    def test_rms_drop_env(self):
        test_bed = [
            ('    ', False),
            ('', False),
            (None, False),
            ('SOME_VAL', True),
        ]
        for val, carry_over in test_bed:
            with TestAreaContext('test_drop_path'):
                # Setup RMS project
                with open('rms_config.yml', 'w') as f:
                    json.dump({
                            'executable': os.path.realpath('bin/rms'),
                        }, f)

                with open('rms_exec_env.json', 'w') as f:
                    json.dump({
                        'RMS_TEST_VAR': val,
                    }, f)
                os.environ['RMS_TEST_VAR'] = 'fdsgfdgfdsgfds'

                os.mkdir("run_path")
                os.mkdir("bin")
                os.mkdir("project")
                shutil.copy(os.path.join(self.SOURCE_ROOT, "python/tests/res/fm/rms"), "bin")
                os.environ["RMS_SITE_CONFIG"] = "rms_config.yml"

                action = {"exit_status" : 0}
                with open("run_path/action.json", "w") as f:
                    f.write( json.dumps(action) )

                rms_exec = os.path.join(self.SOURCE_ROOT, 'share/ert/forward-models/res/script/rms')
                subprocess.check_call([
                    rms_exec,
                    '--run-path',
                    'run_path',
                    '0',
                    '--version',
                    '10.4',
                    'project',
                    '--import-path',
                    './',
                    '--export-path',
                    './',
                    'workflow',
                ])

                with open('run_path/env.json') as f:
                    env = json.load(f)

                if carry_over:
                    self.assertIn('RMS_TEST_VAR', env)
                else:
                    self.assertNotIn('RMS_TEST_VAR', env)



if __name__ == "__main__":
    unittest.main()
