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


class JobManagerTestRuntimeKW(TestCase):
  def test1(self):
        with TestAreaContext("job_manager_runtime_int_kw"):

            executable = "echo"
            job_0 = {"name" : "JOB_1",
                        "executable" : executable,
                        "stdout"    : "outfile.stdout.1",
                        "stderr"    : None,
                        "argList"   : ['a_file', '5.12'],
                        "min_arg"   : 1,
                        "max_arg"   : 2,
                        "arg_types" : ['STRING', 'RUNTIME_INT'] } 

            data = {"umask"              : "0000",
                    "DATA_ROOT"          : "/path/to/data",
                    "jobList"            : [job_0]}

            jobs_file = os.path.join(os.getcwd(), "jobs.json")
            with open(jobs_file, "w") as f:
               f.write(json.dumps(data))
            jobm = JobManager()
            exit_status, msg = jobm.runJob(jobm[0])
            self.assertEqual(exit_status, 1)

  def test2(self):
        with TestAreaContext("job_manager_runtime_file_kw"):
            with open("a_file", "w") as f:
               f.write("Hello")

            executable = "echo"
            job_1 = {"name" : "JOB_0",
                        "executable" : executable,
                        "stdout"    : "outfile.stdout.0",
                        "stderr"    : None,
                        "argList"   : ['some_file'],
                        "min_arg"   : 1,
                        "max_arg"   : 1,
                        "arg_types" : ['RUNTIME_FILE'] }  

            job_0 = {"name" : "JOB_1",
                        "executable" : executable,
                        "stdout"    : "outfile.stdout.1",
                        "stderr"    : None,
                        "argList"   : ['5', 'a_file'],
                        "min_arg"   : 1,
                        "max_arg"   : 2,
                        "arg_types" : ['RUNTIME_INT', 'RUNTIME_FILE'] } 

            data = {"umask"              : "0000",
                    "DATA_ROOT"          : "/path/to/data",
                    "jobList"            : [job_0, job_1]}

            jobs_file = os.path.join(os.getcwd(), "jobs.json")
            with open(jobs_file, "w") as f:
               f.write(json.dumps(data))
            jobm = JobManager()
            exit_status, msg = jobm.runJob(jobm[0])
            self.assertEqual(exit_status, 0)
            exit_status, msg = jobm.runJob(jobm[1])
            self.assertEqual(exit_status, 1)
