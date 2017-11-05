import json
import os
import os.path
import stat
import time
import datetime
import subprocess
from unittest import TestCase

from ecl.test import TestAreaContext
from res.job_queue import JobManager

def gen_area_name(base, f):
    return base + "_" + f.func_name.split("_")[-1]

def create_jobs_py(jobList):
    jobs_file = os.path.join(os.getcwd(), "jobs.py")
    compiled_jobs_file = jobs_file + "c"

    for fname in [jobs_file, compiled_jobs_file]:
        if os.path.isfile(fname):
            os.unlink(fname)

    with open(jobs_file, "w") as f:
        f.write("jobList = ")
        f.write(json.dumps(jobList))
        f.write("\n")

    return jobs_file

def create_jobs_json(jobList, umask="0000"):
    data = {"umask"              : umask,
            "global_environment" : {"FIRST" : "FirstValue" },
            "DATA_ROOT"          : "/path/to/data",
            "jobList"            : jobList}

    jobs_file = os.path.join(os.getcwd(), "jobs.json")
    with open(jobs_file, "w") as f:
        f.write(json.dumps(data))

class JobManagerGetEnvTest(TestCase):
  
    def setUp(self):
        self.dispatch_imp = None
        if "DATA_ROOT" in os.environ:
            del os.environ["DATA_ROOT"]

    def test_get_env(self):
        with TestAreaContext("job_manager_get_env"):
            #with open("x.py", "w") as f:
            #    f.write("#!/usr/bin/env python\n")
            #    f.write("import os\n")
            #    f.write("print('Hello World!')\n")
            #    f.write("print(os.environ['TARGET'])\n")
            #    f.write("print(os.environ['ZXCVBN'])\n")
            #os.chmod("x.py", stat.S_IEXEC + stat.S_IREAD)

            executable = "ls"

            joblist = {"name" : "TEST_GET_ENV",
                        "executable" : executable,
                        "stdout" : "outfile.stdout",
                        "stderr" : "outfile.stderr",
                        "environment" : {"TARGET" : "flatland", "ZXCVBN" : "zxcvbn"},
                        "max_running_minutes" : 12,
                        "max_running" : 30,
                        "argList" : ["-l"] }

            data = {"umask"              : "0000",
                    "global_environment" : {"FIRST" : "FirstValue" },
                    "DATA_ROOT"          : "/path/to/data",
                    "jobList"            : [joblist]}

            jobs_file = os.path.join(os.getcwd(), "jobs.json")
            with open(jobs_file, "w") as f:
               f.write(json.dumps(data))


            jobm = JobManager()
            jobm.runJob(jobm[0])
            subprocess.call(["more", "outfile.stdout.0"])






