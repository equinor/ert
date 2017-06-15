#  Copyright (C) 2017  Statoil ASA, Norway.
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
import os.path
import random
from datetime import datetime as dt
import time
import subprocess
import socket
import pwd
import requests
import json
import imp

LOG_URL = "http://10.220.65.22:4444" #To be extracted up to job_discpatch in the future after a bit of testing and when job_dispatch is properly
#versioned


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
    if err_msg is None:
        err_msg = "JSON-file did not contain a %s." % key
    if key not in data:
        raise IOError(err_msg)
    return data[key]


class JobManager(object):
    LOG_file      = "JOB_LOG"
    EXIT_file     = "ERROR"
    STATUS_file   = "STATUS"
    OK_file       = "OK"

    DEFAULT_UMASK =  0
    sleep_time    =  10  # Time to sleep before exiting the script - to let the disks sync up.



    def __init__(self, module_file="jobs.py", json_file="jobs.json", error_url=None):
        self._job_map = {}
        self._error_url = error_url
        if json_file is not None and os.path.isfile(json_file):
            self._loadJson(json_file)
        else:
            self._loadModule(module_file)

        self.start_time = dt.now()
        ((self.file_server, self.isilon_node), self.fs_use) = JobManager.fsInfo()
        self.max_runtime = 0  # This option is currently sleeping
        self.short_sleep = 2  # Sleep between status checks
        self.node = socket.gethostname()

        pw_entry = pwd.getpwuid(os.getuid())
        self.user = pw_entry.pw_name


        cond_unlink("EXIT")
        cond_unlink(self.EXIT_file)
        cond_unlink(self.STATUS_file)
        cond_unlink(self.OK_file)
        self.initStatusFile()




    def _loadJson(self, json_file_name):
        try:
            with open(json_file_name, "r") as json_file:
                jobs_data = json.load(json_file)
        except ValueError as e:
            raise IOError("Job Manager failed to load JSON-file." + str(e))

        umask = _jsonGet(jobs_data, "umask")
        os.umask(int(umask, 8))

        self.job_list = _jsonGet(jobs_data, "jobList")
        self._ensureCompatibleJobList()
        self._buildJobMap()

    def _loadModule(self, module_file):
        if module_file is None:
            self.job_list = []
            return

        try:
            jobs_module = imp.load_source("jobs", module_file)
        except SyntaxError:
            raise ImportError

        # The internalization of the job items is currently *EXTREMELY* basic.
        self.job_list = jobs_module.jobList
        self._ensureCompatibleJobList()
        self._buildJobMap()

        if hasattr(jobs_module, "umask"):
            umask = jobs_module.umask
        else:
            umask = self.DEFAULT_UMASK
        os.umask(umask)

    # To ensure compatibility with old versions.
    def _ensureCompatibleJobList(self):
        for job in self.job_list:
            if not job.has_key("max_running_minutes"):
                job["max_running_minutes"] = None

    def _buildJobMap(self):
        self._job_map = {}
        for index, job in enumerate(self.job_list):
            self._job_map[job["name"]] = job
            if "stderr" in job:
                job["stderr"] = "%s.%d" % (job["stderr"], index)

            if "stdout" in job:
                job["stdout"] = "%s.%d" % (job["stdout"], index)

    def __contains__(self, key):
        return key in self._job_map


    def __len__(self):
        return len(self.job_list)


    def __getitem__(self, index):
        if isinstance(index, int):
            return self.job_list[index]
        else:
            return self._job_map[index]


    def initStatusFile(self):
        with open(self.STATUS_file, "a") as f:
            f.write("%-32s: %s/%s  file-server:%s \n" % ("Current host", self.node, os.uname()[4], self.isilon_node))


    def startStatus(self, job):
        with open(self.STATUS_file, "a") as f:
            now = time.localtime()
            f.write("%-32s: %02d:%02d:%02d .... " % (job["name"], now.tm_hour, now.tm_min, now.tm_sec))
        self.postMessage(job=job)


    def completeStatus(self, exit_status, error_msg):
        now = time.localtime()
        extra_fields = {"finished": True,
                        "exit_status": exit_status,
                        "status": "completeStatus"}
        with open(self.STATUS_file, "a") as f:
            if exit_status == 0:
                status = ""
                self.postMessage(extra_fields=extra_fields)
            else:
                status = " EXIT: %d/%s" % (exit_status, error_msg)
                extra_fields.update({"error_msg": error_msg})
                self.postMessage(extra_fields=extra_fields)

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


    def getFileServer(self):
        return self.isilon_node


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


    def postMessage(self, job=None, extra_fields={}, url=LOG_URL):
        if job:
            job_fields = {"ert_job": job["name"],
                           "executable": job["executable"],
                           "arg_list": " ".join(job["argList"])}
            job_fields.update(extra_fields)
            extra_fields = job_fields

        payload = {"user": self.user,
                   "cwd": os.getcwd(),
                   "file_server": self.isilon_node,
                   "node": self.node,
                   "start_time": self.start_time.isoformat(),
                   "fs_use": "%s / %s / %s" % self.fs_use,
                   "fs_utilization": "%s" % (self.fs_use[2])[:-1], #remove the "%"
                   "node_timestamp": dt.now().isoformat()}
        payload.update(extra_fields)
        try:
            if url is None:
                sys.stderr.write('\nWARNING: LOG/ERROR URL NOT CONFIGURED.\n\n')
                sys.stderr.write(json.dumps(payload))
                sys.stderr.write('\nAbove error log NOT submitted.')
                sys.stderr.flush()
            else:
                data = json.dumps(payload)
                res = requests.post(url, timeout=3,
                              headers={"Content-Type": "application/json"},
                              data=data)
                sys.stdout.write("Response status %s\n"%res.status_code)
                sys.stdout.write("Request url %s\n"%res.url)
                sys.stdout.write("Response headers %s\n"%res.headers)
                sys.stdout.write("Response content %s\n"%res.content)
                sys.stdout.write("Writing payload: %s\n"%payload)
                sys.stdout.write("Writing data: %s\n"%data)
        except:
            pass

    def postError(self, job, error_msg):
        extra_fields = self.extract_stderr_stdout(job)
        self.postMessage(job, extra_fields, url=self._error_url)

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
        self.postMessage(job=job, extra_fields=std_err_out, url=self._error_url) #posts to the old database
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
        self.postMessage(job=job, extra_fields={"status": "run","finished": False})
        pid = os.fork()
        exit_status, err_msg = 0, ''
        if pid == 0:
            self.execJob(job)
        else:
            _, exit_status = os.waitpid(pid, 0)
            # The exit_status returned from os.waitpid encodes
            # both the exit status of the external application,
            # and in case the job was killed by a signal - the
            # number of that signal.
            exit_status = os.WEXITSTATUS(exit_status)
        if exit_status != 0:
            err_msg = "Executable: %s failed with exit code: %s" % (job.get('executable'),
                                                                    exit_status)

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

    @staticmethod
    def fsInfo(path = None):
        if path is None:
            path = os.getcwd()

        if not os.path.isabs(path):
            raise ValueError("Must have an absolute path")

        if not os.path.exists(path):
            raise ValueError("No such entry: %s" % path)

        path_list = path.split("/")
        if len(path_list) < 3:
            raise ValueError("Must have at least two levels in directory name")

        mount_point = "/%s/%s" % (path_list[1], path_list[2])
        file_server, isilon_node = JobManager.mountPoint(mount_point)

        df_stdout = subprocess.check_output(["df", "-Ph", path]).strip().split('\n')
        line1 = df_stdout[1].split()
        size = line1[1]
        free = line1[3]
        util = line1[4]
        return ((file_server, isilon_node), (size, free, util))


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
