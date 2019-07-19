from tests import ResTest
from res.enkf import ResConfig
from ecl.util.test import TestAreaContext

import os, stat
import json
from subprocess import Popen
import time
import psutil

class JobDispatchTest(ResTest):

    def testTerminateJobs(self):

        with TestAreaContext("job_dispatch_test") as work_area:
            
            # Executes it self recursively and sleeps for 100 seconds
            with open("dummy_executable", "w") as f:
                f.write("#!/usr/bin/env python\n"\
                        "import sys, os, time\n"\
                        "counter = eval(sys.argv[1])\n"\
                        "if counter > 0:\n"\
                        "    os.fork()\n"\
                        "    os.execv(sys.argv[0],[sys.argv[0], str(counter - 1) ])\n"\
                        "else:\n"\
                        "    time.sleep(100)\n"
                    )
                    
            executable = os.path.realpath("dummy_executable")
            os.chmod("dummy_executable", stat.S_IRWXU |  stat.S_IRWXO |  stat.S_IRWXG )
            
            self.job_list = {
                "umask" : "0002",
                "DATA_ROOT": "",
                "global_environment" : {},
                "global_update_path" : {},
                "jobList" : [ {
                    "name" : "dummy_executable",
                    "executable" : executable,
                    "target_file" : None,
                    "error_file" : None,
                    "start_file" : None,
                    "stdout" : "dummy.stdout",
                    "stderr" : "dummy.stderr",
                    "stdin" : None,
                    "argList" : ['3'],
                    "environment" : None,
                    "exec_env" : None,
                    "license_path" : None,
                    "max_running_minutes" : None,
                    "max_running" : None,
                    "min_arg" : 1,
                    "arg_types" : [],
                    "max_arg" : None
                }],
                "ert_version" : [],
                "run_id" : "",
                "ert_pid" : ""
                }

            job_dispatch_script = os.path.realpath(
                os.path.join(
                    os.path.dirname(
                        os.path.abspath(__file__)
                        ),
                        '../../../../bin/job_dispatch.py'
                    )
                )
            
            with open("jobs.json", "w") as f:
                f.write(json.dumps(self.job_list))

            # Required to execute job_dispatch in separate process group by 
            # os.setsid moves the current process out of the current group
            with open("job_dispatch_executer",'w') as f:
                f.write("#!/usr/bin/env python\n"\
                        "import os, sys\n"\
                        "os.setsid()\n"
                        "os.execv(sys.argv[1], sys.argv[1:])\n"\
                        "\n")
            os.chmod("job_dispatch_executer", stat.S_IRWXU |  stat.S_IRWXO |  stat.S_IRWXG )

            current_dir = os.path.realpath(os.curdir)
            job_dispatch_process = Popen([os.path.realpath("job_dispatch_executer"), job_dispatch_script, current_dir])
            
            time.sleep(1) # wait for processes to be spawned
            p = psutil.Process(job_dispatch_process.pid)
            
            # Three levels of processes should spawn 8 children in total
            self.assertEqual(len(p.children(recursive=True)), 8)
            
            p.terminate()
            time.sleep(1) # wait for processes to be terminated
            self.assertEqual(len(p.children(recursive=True)), 0)
            
            os.wait() # allow os to clean up zombie processes
