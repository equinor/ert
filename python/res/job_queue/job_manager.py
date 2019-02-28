#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  This file is part of ERT - Ensemble based Reservoir Tool.
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

import sys
import os
import signal
import shutil
import random
from datetime import datetime as dt
import time
import subprocess
import socket
import pwd
import requests
import json
import imp
from ecl import EclVersion
from res import ResVersion
from res.job_queue import ForwardModelStatus, ForwardModelJobStatus
from sys import version as sys_version

def redirect(file, fd, open_mode):
    new_fd = os.open(file, open_mode)
    os.dup2(new_fd, fd)
    os.close(new_fd)


def redirect_input(file, fd):
    redirect(file, fd, os.O_RDONLY)


def redirect_output(file, fd,start_time):
    if os.path.isfile(file):
        mtime = os.path.getmtime(file)
        if mtime < start_time:
            # Old stale version; truncate.
            redirect(file, fd, os.O_WRONLY | os.O_TRUNC | os.O_CREAT)
        else:
            # A new invocation of the same job instance; append putput
            redirect(file, fd, os.O_APPEND)
    else:
        redirect(file, fd, os.O_WRONLY | os.O_TRUNC | os.O_CREAT)


def cond_unlink(file):
    if os.path.exists(file):
        os.unlink(file)


def assert_file_executable(fname):
    """The function raises an IOError if the given file is either not a file or
    not an executable.

    If the given file name is an absolute path, its functionality is straight
    forward. When given a relative path it will look for the given file in the
    current directory as well as all locations specified by the environment
    path.

    """
    if not fname:
        raise IOError('No executable provided!')
    fname = os.path.expanduser(fname)

    potential_executables = [os.path.abspath(fname)]
    if not os.path.isabs(fname):
        potential_executables = (potential_executables +
            [
                os.path.join(location, fname)
                for location in os.environ["PATH"].split(os.pathsep)
            ])

    if not any(map(os.path.isfile, potential_executables)):
        raise IOError("%s is not a file!" %fname)

    if not any([os.access(fn, os.X_OK) for fn in potential_executables]):
        raise IOError("%s is not an executable!" %fname)


def _jsonGet(data, key, err_msg=None):
    if key not in data:
        if err_msg is None:
            err_msg = "JSON-file did not contain a %s." % key
        raise IOError(err_msg)
    return data[key]

def _read_os_release(pfx='LSB_'):
    fname = '/etc/os-release'
    if not os.path.isfile(fname):
        return {}
    def processline(ln):
        return ln.strip().replace('"', '')
    def splitline(ln, pfx=''):
        if ln.count('=') == 1:
            k, v = ln.split('=')
            return pfx+k, v
        return None
    props = {}
    with open(fname, 'r') as f:
        for line in f:
            kv = splitline(processline(line), pfx=pfx)
            if kv:
                props[kv[0]] = kv[1]
    return props

def pad_nonexisting(path, pad='-- '):
    return path if os.path.exists(path) else pad + path

class JobManager(object):
    LOG_file      = "JOB_LOG"
    EXIT_file     = "ERROR"
    STATUS_file   = "STATUS"
    OK_file       = "OK"

    DEFAULT_UMASK =  0
    sleep_time    =  10  # Time to sleep before exiting the script - to let the disks sync up.



    def __init__(self, module_file="jobs.py", json_file="jobs.json", error_url=None, log_url=None):
        self._job_map = {}
        self.simulation_id = ""
        self.ert_pid = ""
        self._log_url = log_url
        if log_url is None:
            self._log_url = error_url
        self._data_root = None
        self.global_environment = None
        self.global_update_path = None
        self.start_time = dt.now()
        if json_file is not None and os.path.isfile(json_file):
            self.job_status = ForwardModelStatus("????", self.start_time)
            self._loadJson(json_file)
            self.job_status.run_id = self.simulation_id
        else:
            raise IOError("'jobs.json' not found.")

        self.max_runtime = 0  # This option is currently sleeping
        self.short_sleep = 2  # Sleep between status checks
        self.node = socket.gethostname()
        pw_entry = pwd.getpwuid(os.getuid())
        self.user = pw_entry.pw_name
        os_info = _read_os_release()
        _, _, release, _, _ = os.uname()
        python_vs, _ = sys_version.split('\n')
        ecl_v = EclVersion()
        res_v = ResVersion()
        logged_fields= {"status": "init",
                        "python_sys_path": map(pad_nonexisting, sys.path),
                        "pythonpath": map(pad_nonexisting, os.environ.get('PYTHONPATH', '').split(':')),
                        "res_version": res_v.versionString(),
                        "ecl_version": ecl_v.versionString(),
                        "LSB_ID": os_info.get('LSB_ID', ''),
                        "LSB_VERSION_ID": os_info.get('LSB_VERSION_ID', ''),
                        "python_version": python_vs,
                        "kernel_version": release,
                        }
        logged_fields.update({"jobs": self._ordered_job_map_values()})
        self.postMessage(extra_fields=logged_fields)
        cond_unlink("EXIT")
        cond_unlink(self.EXIT_file)
        cond_unlink(self.STATUS_file)
        cond_unlink(self.OK_file)
        self.initStatusFile()
        if self._data_root:
            os.environ["DATA_ROOT"] = self._data_root
        self.set_environment()
        self.update_path()
        self.information = logged_fields


    def complete(self):
        self.job_status.complete()

    def dump_status(self):
        self.job_status.dump(ForwardModelStatus.STATUS_FILE)

    def set_environment(self):
         if self.global_environment:
           data = self.global_environment
           for key in data.keys():
               os.environ[key] = data[key]

    def update_path(self):
        if self.global_update_path:
           data = self.global_update_path
           for key in data.keys():
               if (os.environ.get(key)):
                  os.environ[key] = data[key] + ':' + os.environ[key]
               else:
                  os.environ[key] = data[key]


    def data_root(self):
        return self._data_root


    def _loadJson(self, json_file_name):
        try:
            with open(json_file_name, "r") as json_file:
                jobs_data = json.load(json_file)
        except ValueError as e:
            raise IOError("Job Manager failed to load JSON-file." + str(e))

        self._data_root = jobs_data.get("DATA_ROOT")
        umask = _jsonGet(jobs_data, "umask")
        os.umask(int(umask, 8))
        if "run_id" in jobs_data:
            self.simulation_id = _jsonGet(jobs_data, "run_id")
            os.environ["ERT_RUN_ID"] = self.simulation_id
        if "ert_pid" in jobs_data:
            self.ert_pid = _jsonGet(jobs_data, "ert_pid")
        if "global_environment" in jobs_data:
            self.global_environment = _jsonGet(jobs_data, "global_environment")
        if "global_update_path" in jobs_data:
            self.global_update_path = _jsonGet(jobs_data, "global_update_path")
        self.job_list = _jsonGet(jobs_data, "jobList")
        self._ensureCompatibleJobList()
        self._buildJobMap()

        for job in self.job_list:
            self.job_status.add_job( ForwardModelJobStatus(job.get("name")))

        # "Monkey-patching" the job object by attaching a status object.
        status_list = self.job_status.jobs
        for i in range(len(self.job_list)):
            self.job_list[i]["status"] = status_list[i]

    # To ensure compatibility with old versions.
    def _ensureCompatibleJobList(self):
        for job in self.job_list:
            if not "max_running_minutes" in job.keys():
                job["max_running_minutes"] = None

    def _buildJobMap(self):
        self._job_map = {}
        for index, job in enumerate(self.job_list):
            self._job_map[job["name"]] = job
            if "stderr" in job:
                if job["stderr"]:
                   job["stderr"] = "%s.%d" % (job["stderr"], index)

            if "stdout" in job:
                if job["stdout"]:
                   job["stdout"] = "%s.%d" % (job["stdout"], index)

    def _ordered_job_map_values(self):
        ordered_map_values = []
        for index, job in enumerate(self.job_list):
            ordered_map_values.append(self._job_map[job["name"]])
        return ordered_map_values

    def __contains__(self, key):
        return key in self._job_map

    def __len__(self):
        return len(self.job_list)

    def __repr__(self):
        st = self.start_time
        node = self.node
        us = self.user
        cnt = 'len=%d, start=%s, node=%s, user=%s'
        cnt = cnt % (len(self), st, node, us)
        return 'JobManager(%s)' % cnt


    def __getitem__(self, index):
        if isinstance(index, int):
            return self.job_list[index]
        else:
            return self._job_map[index]


    def initStatusFile(self):
        with open(self.STATUS_file, "a") as f:
            f.write("%-32s: %s/%s\n" % ("Current host", self.node, os.uname()[4]))

    def startStatus(self, job):
        with open(self.STATUS_file, "a") as f:
            now = time.localtime()
            f.write("%-32s: %02d:%02d:%02d .... " % (job["name"], now.tm_hour, now.tm_min, now.tm_sec))


    def completeStatus(self, exit_status, error_msg, job=None):
        now = time.localtime()
        extra_fields = {"finished": True,
                        "exit_status": exit_status,
                        "status": "completeStatus"}
        with open(self.STATUS_file, "a") as f:
            if exit_status == 0:
                status = ""
            else:
                status = " EXIT: %d/%s" % (exit_status, error_msg)
                extra_fields.update({"error_msg": error_msg})

            f.write("%02d:%02d:%02d  %s\n" % (now.tm_hour, now.tm_min, now.tm_sec, status))


    def createOKFile(self):
        now = time.localtime()
        with open(self.OK_file, "w") as f:
            f.write("All jobs complete %02d:%02d:%02d \n" % (now.tm_hour, now.tm_min, now.tm_sec))
        self.postMessage(extra_fields={"status" : "OK"})
        time.sleep(self.sleep_time)   # Let the disks sync up


    def getStartTime(self):
        return self.start_time


    def getRuntime(self):
        rt = dt.now() - self.start_time
        return rt.total_seconds()


    def assertArgList(self, job):
        if "arg_types" in job:
            argTypes = job["arg_types"]
            argList = job.get("argList")
            for index, arg_type in enumerate(argTypes):
                if (arg_type == "RUNTIME_FILE"):
                    file_path = os.path.join(os.getcwd(), argList[index])
                    if not os.path.isfile(file_path):
                        raise TypeError("In job \"%s\": RUNTIME_FILE \"%s\" does not exist." % (job["name"], argList[index]))
                if (arg_type == "RUNTIME_INT"):
                    try:
                        int(argList[index])
                    except ValueError:
                        raise ValueError("In job \"%s\": argument with index %d is of incorrect type, should be integer." % (job["name"], index))


    def execJob(self, job):
        executable = job.get('executable')
        assert_file_executable(executable)

        start_time = time.time()
        if job.get("stdin"):
            redirect_input(job["stdin"] , 0)

        if job.get("stdout"):
            redirect_output(job["stdout"], 1, start_time)

        if job.get("stderr"):
            redirect_output(job["stderr"], 2, start_time)

        if job.get("environment"):
            env = job["environment"]
            for key in env.keys():
                os.putenv(key, env[key])

        self.assertArgList(job)

        argList = [executable]
        if job.get("argList"):
            argList += job["argList"]

        os.execvp(executable, argList)


    def jobProcess(self, job):
        executable = job.get('executable')
        assert_file_executable(executable)

        argList = [executable]
        if job.get("argList"):
            argList += job["argList"]

        if job.get("stdin"):
            stdin = open(job.get("stdin"))
        else:
            stdin = None

        if job.get("stderr"):
            stderr = open(job.get("stderr"), "w")
        else:
            stderr = None

        if job.get("stdout"):
            stdout = open(job.get("stdout"), "w")
        else:
            stdout = None

        P = subprocess.Popen(argList,
                             stdin=stdin,
                             stdout=stdout,
                             stderr=stderr,
                             env=job.get("environment"))

        return P


    def postMessage(self, job=None, extra_fields={}, url=None):
        if url is None:
            url=self._log_url
        if job:
            job_fields = {"ert_job": job["name"],
                           "executable": job["executable"],
                           "arg_list": " ".join(job["argList"])}
            job_fields.update(extra_fields)
            extra_fields = job_fields

        payload = {"user": self.user,
                   "cwd": os.getcwd(),
                   "application": "ert",
                   "subsystem": "ert_forward_model",
                   "node": self.node,
                   "komodo_release": os.getenv("KOMODO_RELEASE", "--------"),
                   "start_time": self.start_time.isoformat(),
                   "node_timestamp": dt.now().isoformat(),
                   "simulation_id": self.simulation_id,
                   "ert_pid": self.ert_pid}
        payload.update(extra_fields)
        try:
            if url is None:
                sys.stderr.write('\nWARNING: LOG/ERROR URL NOT CONFIGURED.\n\n')
                sys.stderr.write(json.dumps(payload))
                sys.stderr.write('\nAbove error log NOT submitted.')
                sys.stderr.flush()
            else:
                data = json.dumps(payload)
                #Disabling proxies
                proxies = {
                    "http": None,
                    "https": None,
                }
                res = requests.post(url, timeout=3,
                              headers={"Content-Type": "application/json"},
                                    data=data,proxies=proxies)
                # sys.stdout.write("Response status %s\n"%res.status_code)
                # sys.stdout.write("Request url %s\n"%res.url)
                # sys.stdout.write("Response headers %s\n"%res.headers)
                # sys.stdout.write("Response content %s\n"%res.content)
                # sys.stdout.write("Writing payload: %s\n"%payload)
                # sys.stdout.write("Writing data: %s\n"%data)
        except:
            pass

    def postError(self, job, error_msg):
        extra_fields = self.extract_stderr_stdout(job)
        extra_fields.update({"status": "error","finished": True})
        self.postMessage(job, extra_fields, url=self._log_url)

    def extract_stderr_stdout(self, job):
        extra_fields = {}
        if job.get("stderr"):
            if os.path.exists(job["stderr"]):
                with open(job["stderr"], "r") as errH:
                    stderr = errH.read()
                    extra_fields.update({"stderr": stderr})
        if job.get("stdout"):
            if os.path.exists(job["stdout"]):
                with open(job["stdout"], "r") as outH:
                    stdout = outH.read()
                    extra_fields.update({"stdout": stdout})
        return extra_fields

    def exit(self, job, exit_status, error_msg):
        self.dump_EXIT_file(job, error_msg)
        std_err_out = self.extract_stderr_stdout(job)
        std_err_out.update({"status": "exit","finished": True, "error_msg": error_msg, "exit_status": exit_status, "error": True})
        self.postMessage(job=job, extra_fields=std_err_out) #Posts to new logstash
        pgid = os.getpgid(os.getpid())
        os.killpg(pgid, signal.SIGKILL)



    def addLogLine(self, job):
        now = time.localtime()
        with open(self.LOG_file, "a") as f:
            args = " ".join(job["argList"])
            f.write("%02d:%02d:%02d  Calling: %s %s\n" %
                    (now.tm_hour, now.tm_min, now.tm_sec,
                     job.get('executable'), args))

    def runJob(self, job):
        assert_file_executable(job.get('executable'))
        self.addLogLine(job)

        status = job["status"]
        status.start_time = dt.now()
        status.status = "Running"
        self.job_status.dump()

        exec_env = job.get("exec_env")
        if exec_env:
            exec_name,_ = os.path.splitext(os.path.basename(job.get('executable')))
            with open("%s_exec_env.json" % exec_name, "w") as f:
                f.write(json.dumps(exec_env))

        pid = os.fork()
        exit_status, err_msg = 0, ''
        if pid == 0:
            # This code block should exec into the actual executable we are
            # running, and execution should not come back here. However - if
            # the code fails with an exception before actually reaching the
            # exec() call we suddenly have two Python processes running the
            # current code; one waiting for the exit status and one unrolling
            # an exception. The latter will incorrectly "steal" the
            # finalization of with statements. So - in the case of an exception
            # before the exec() call we call the hard exit: os._exit(1).
            try:
                self.execJob(job)
            except Exception as e:
                sys.stderr.write("Failed to exec:%s error:%s\n" % (job["name"], str(e)))
                os._exit(1)
        else:
            _, exit_status = os.waitpid(pid, 0)
            # The exit_status returned from os.waitpid encodes
            # both the exit status of the external application,
            # and in case the job was killed by a signal - the
            # number of that signal.
            exit_status = os.WEXITSTATUS(exit_status)

        status.end_time = dt.now()

        if exit_status != 0:
            err_msg = "Executable: %s failed with exit code: %s" % (job.get('executable'),
                                                                    exit_status)

            status.status = "Failure"
            status.error = err_msg
        else:
            status.status = "Success"

        self.job_status.dump()
        return exit_status, err_msg



    @staticmethod
    def mountPoint(path):
        """Calls `mount`, finds line corresponding to given path, returns addr part."""
        mount_stdout = subprocess.check_output(["mount"]).strip().split('\n')
        for line in mount_stdout:
            tmp = line.split()
            if tmp[2] == path:
                cnt = tmp[5][1:-1] # was '(rw,...,addr=...)' is now 'rw,...,addr=...,'
                d = dict([tuple(x.split('=')) if '=' in x else (x, True) for x in cnt.split(',')])
                if 'addr' in d:
                    isilon_node = d['addr']
                elif 'mountaddr' in d:
                    isilon_node = d['mountaddr']

                server_tmp = tmp[0].split(":")
                if len(server_tmp) == 1:
                    file_server = "local"
                else:
                    file_server = server_tmp[0]

                return (file_server, isilon_node)

        return ('?', '?.?.?.?')


    # This file will be read by the job_queue_node_fscanf_EXIT() function
    # in job_queue.c. Be very careful with changes in output format.
    def dump_EXIT_file(self, job, error_msg):
        fileH = open(self.EXIT_file, "a")
        now = time.localtime()
        fileH.write("<error>\n")
        fileH.write("  <time>%02d:%02d:%02d</time>\n" % (now.tm_hour, now.tm_min, now.tm_sec))
        fileH.write("  <job>%s</job>\n" % job["name"])
        fileH.write("  <reason>%s</reason>\n" % error_msg)
        stderr_file = None
        if job["stderr"]:
            if os.path.exists(job["stderr"]):
                with open(job["stderr"], "r") as errH:
                    stderr = errH.read()
                    if stderr:
                        stderr_file = os.path.join(os.getcwd(), job["stderr"])
                    else:
                        stderr = "<Not written by:%s>\n" % job["name"]
            else:
                stderr = "<stderr: Could not find file:%s>\n" % job["stderr"]
        else:
            stderr = "<stderr: Not redirected>\n"

        fileH.write("  <stderr>\n%s</stderr>\n" % stderr)
        if stderr_file:
            fileH.write("  <stderr_file>%s</stderr_file>\n" % stderr_file)

        fileH.write("</error>\n")
        fileH.close()

        # Have renamed the exit file from "EXIT" to "ERROR";
        # must keep the old "EXIT" file around until all old ert versions
        # are flushed out.
        shutil.copyfile(self.EXIT_file, "EXIT")
