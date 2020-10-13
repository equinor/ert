import json
import os
import pwd
import socket
import sys

# shadow map builtin in py2, making our map usage py2 compatible
# TODO: safely remove after py2 support in ERT ends
from builtins import map
from sys import version as sys_version

import requests
from ecl import EclVersion
from job_runner.reporting.message import Exited, Init, Finish
from job_runner.util import pad_nonexisting, read_os_release
from res import ResVersion
from ert_logger import log_message


class Network(object):
    def __init__(self):
        self.simulation_id = None
        self.ert_pid = None
        self.start_time = None
        self.node = socket.gethostname()

        pw_entry = pwd.getpwuid(os.getuid())
        self.user = pw_entry.pw_name

    def report(self, msg):
        if isinstance(msg, Init):
            self.start_time = msg.timestamp
            self.simulation_id = msg.run_id
            self.ert_pid = msg.ert_pid

            self._post_initial(msg)
        elif isinstance(msg, Exited):
            if not msg.success():
                self._post_job_failure(msg)
        elif isinstance(msg, Finish):
            if msg.success():
                self._post_success(msg)

    def _post_initial(self, msg):
        os_info = read_os_release()
        _, _, release, _, _ = os.uname()
        python_vs, _ = sys_version.split("\n")
        ecl_v = EclVersion()
        res_v = ResVersion()
        logged_fields = {
            "status": "init",
            "python_sys_path": list(map(pad_nonexisting, sys.path)),
            "pythonpath": list(
                map(pad_nonexisting, os.environ.get("PYTHONPATH", "").split(":"))
            ),
            "res_version": res_v.versionString(),
            "ecl_version": ecl_v.versionString(),
            "LSB_ID": os_info.get("LSB_ID", ""),
            "LSB_VERSION_ID": os_info.get("LSB_VERSION_ID", ""),
            "python_version": python_vs,
            "kernel_version": release,
        }

        job_list = [j.name() for j in msg.jobs]
        logged_fields.update({"jobs": job_list})

        self._post_message(msg.timestamp, extra_fields=logged_fields)

    def _post_message(self, timestamp, extra_fields=None):
        payload = {
            "cwd": os.getcwd(),
            "subsystem": "ert_forward_model",
            "node": self.node,
            "start_time": self.start_time.isoformat(),
            "node_timestamp": timestamp.isoformat(),
            "simulation_id": self.simulation_id,
            "ert_pid": self.ert_pid,
        }
        payload.update(extra_fields)
        log_message(payload)

    def _post_success(self, msg):
        self._post_message(msg.timestamp, {"status": "OK"})

    def _post_job_failure(self, msg):
        fields = {
            "status": "exit",
            "exit_status": msg.exit_code,
            "finished": True,
            "error": True,
            "error_msg": msg.error_message,
            "ert_job": msg.job.name(),
            "executable": msg.job.job_data["executable"],
            "arg_list": " ".join(msg.job.job_data["argList"]),
        }
        fields.update(self._extract_stderr_stdout(msg.job))
        self._post_message(msg.timestamp, fields)

    def _extract_stderr_stdout(self, job):
        extra_fields = {}
        if job.std_err:
            if os.path.exists(job.std_err):
                with open(job.std_err, "r") as errH:
                    stderr = errH.read()
                    extra_fields.update({"stderr": stderr})
        if job.std_out:
            if os.path.exists(job.std_out):
                with open(job.std_out, "r") as outH:
                    stdout = outH.read()
                    extra_fields.update({"stdout": stdout})

        return extra_fields
