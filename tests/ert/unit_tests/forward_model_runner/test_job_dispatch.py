from __future__ import annotations

import glob
import importlib
import json
import os
import signal
import stat
import subprocess
import sys
from subprocess import Popen
from textwrap import dedent
from threading import Lock
from unittest.mock import mock_open, patch

import pandas as pd
import psutil
import pytest

import _ert.forward_model_runner.cli
from _ert.forward_model_runner.cli import JOBS_FILE, _setup_reporters, main
from _ert.forward_model_runner.job import killed_by_oom
from _ert.forward_model_runner.reporting import Event, Interactive
from _ert.forward_model_runner.reporting.message import Finish, Init
from _ert.threading import ErtThread
from tests.ert.utils import _mock_ws_thread, wait_until

from .test_event_reporter import _wait_until


@pytest.mark.usefixtures("use_tmpdir")
def test_terminate_jobs():
    # Executes it self recursively and sleeps for 100 seconds
    with open("dummy_executable", "w", encoding="utf-8") as f:
        f.write(
            """#!/usr/bin/env python
import sys, os, time
counter = eval(sys.argv[1])
if counter > 0:
    os.fork()
    os.execv(sys.argv[0],[sys.argv[0], str(counter - 1) ])
else:
    time.sleep(100)"""
        )

    executable = os.path.realpath("dummy_executable")
    os.chmod("dummy_executable", stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)

    job_list = {
        "global_environment": {},
        "global_update_path": {},
        "jobList": [
            {
                "name": "dummy_executable",
                "executable": executable,
                "target_file": None,
                "error_file": None,
                "start_file": None,
                "stdout": "dummy.stdout",
                "stderr": "dummy.stderr",
                "stdin": None,
                "argList": ["3"],
                "environment": None,
                "exec_env": None,
                "license_path": None,
                "max_running_minutes": None,
                "min_arg": 1,
                "arg_types": [],
                "max_arg": None,
            }
        ],
        "run_id": "",
        "ert_pid": "",
    }

    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(job_list))

    # macOS doesn't provide /usr/bin/setsid, so we roll our own
    with open("setsid", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
            #!/usr/bin/env python
            import os
            import sys
            os.setsid()
            os.execvp(sys.argv[1], sys.argv[1:])
            """
            )
        )
    os.chmod("setsid", 0o755)

    job_dispatch_script = importlib.util.find_spec(
        "_ert.forward_model_runner.job_dispatch"
    ).origin
    # (we wait for the process below)
    job_dispatch_process = Popen(
        [
            os.getcwd() + "/setsid",
            sys.executable,
            job_dispatch_script,
            os.getcwd(),
        ]
    )

    p = psutil.Process(job_dispatch_process.pid)

    # Three levels of processes should spawn 8 children in total
    wait_until(lambda: len(p.children(recursive=True)) == 8)

    p.terminate()

    wait_until(lambda: len(p.children(recursive=True)) == 0)

    os.wait()  # allow os to clean up zombie processes


@pytest.mark.usefixtures("use_tmpdir")
def test_memory_profile_is_logged_as_csv():
    """This tests that a csv is produced and has basic validity.
    It does not try to verify the validity of the logged RSS values."""
    fm_stepname = "do_nothing"
    scriptname = fm_stepname + ".py"
    fm_step_repeats = 3
    with open(scriptname, "w", encoding="utf-8") as script:
        script.write(
            """#!/bin/sh
        exit 0
        """
        )
    os.chmod(scriptname, stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)
    forward_model_steps = {
        "jobList": [
            {
                "name": fm_stepname,
                "executable": os.path.realpath(scriptname),
                "argList": [""],
            }
        ]
        * fm_step_repeats,
    }

    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(forward_model_steps))

    subprocess.run(
        [
            sys.executable,
            importlib.util.find_spec("_ert.forward_model_runner.job_dispatch").origin,
            os.getcwd(),
        ],
        check=False,
    )
    csv_files = glob.glob("logs/memory-profile*csv")
    mem_df = pd.read_csv(csv_files[0], parse_dates=True)
    assert mem_df["timestamp"].is_monotonic_increasing
    assert (mem_df["fm_step_id"].values == [0, 1, 2]).all()
    assert mem_df["fm_step_name"].unique() == [fm_stepname]
    assert (mem_df["rss"] >= 0).all()  # 0 has been observed


@pytest.mark.usefixtures("use_tmpdir")
def test_job_dispatch_run_subset_specified_as_parameter():
    with open("dummy_executable", "w", encoding="utf-8") as f:
        f.write(
            "#!/usr/bin/env python\n"
            "import sys, os\n"
            'filename = "job_{}.out".format(sys.argv[1])\n'
            'f = open(filename, "w", encoding="utf-8")\n'
            "f.close()\n"
        )

    executable = os.path.realpath("dummy_executable")
    os.chmod("dummy_executable", stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)

    job_list = {
        "global_environment": {},
        "global_update_path": {},
        "jobList": [
            {
                "name": "job_A",
                "executable": executable,
                "target_file": None,
                "error_file": None,
                "start_file": None,
                "stdout": "dummy.stdout",
                "stderr": "dummy.stderr",
                "stdin": None,
                "argList": ["A"],
                "environment": None,
                "exec_env": None,
                "license_path": None,
                "max_running_minutes": None,
                "min_arg": 1,
                "arg_types": [],
                "max_arg": None,
            },
            {
                "name": "job_B",
                "executable": executable,
                "target_file": None,
                "error_file": None,
                "start_file": None,
                "stdout": "dummy.stdout",
                "stderr": "dummy.stderr",
                "stdin": None,
                "argList": ["B"],
                "environment": None,
                "exec_env": None,
                "license_path": None,
                "max_running_minutes": None,
                "min_arg": 1,
                "arg_types": [],
                "max_arg": None,
            },
            {
                "name": "job_C",
                "executable": executable,
                "target_file": None,
                "error_file": None,
                "start_file": None,
                "stdout": "dummy.stdout",
                "stderr": "dummy.stderr",
                "stdin": None,
                "argList": ["C"],
                "environment": None,
                "exec_env": None,
                "license_path": None,
                "max_running_minutes": None,
                "min_arg": 1,
                "arg_types": [],
                "max_arg": None,
            },
        ],
        "run_id": "",
        "ert_pid": "",
    }

    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(job_list))

    # macOS doesn't provide /usr/bin/setsid, so we roll our own
    with open("setsid", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
            #!/usr/bin/env python
            import os
            import sys
            os.setsid()
            os.execvp(sys.argv[1], sys.argv[1:])
            """
            )
        )
    os.chmod("setsid", 0o755)

    job_dispatch_script = importlib.util.find_spec(
        "_ert.forward_model_runner.job_dispatch"
    ).origin
    # (we wait for the process below)
    job_dispatch_process = Popen(
        [
            os.getcwd() + "/setsid",
            sys.executable,
            job_dispatch_script,
            os.getcwd(),
            "job_B",
            "job_C",
        ]
    )

    job_dispatch_process.wait()

    assert not os.path.isfile("job_A.out")
    assert os.path.isfile("job_B.out")
    assert os.path.isfile("job_C.out")


def test_no_jobs_json_file_raises_IOError(tmp_path):
    with pytest.raises(IOError):
        main(["script.py", str(tmp_path)])


def test_invalid_jobs_json_raises_OSError(tmp_path):
    (tmp_path / JOBS_FILE).write_text("not json")

    with pytest.raises(OSError):
        main(["script.py", str(tmp_path)])


def test_missing_directory_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["script.py", str(tmp_path / "non_existent")])


def test_retry_of_jobs_file_read(unused_tcp_port, tmp_path, monkeypatch, caplog):
    lock = Lock()
    lock.acquire()
    monkeypatch.setattr(_ert.forward_model_runner.cli, "_wait_for_retry", lock.acquire)
    jobs_json = json.dumps(
        {
            "ens_id": "_id_",
            "dispatch_url": f"ws://localhost:{unused_tcp_port}",
            "jobList": [],
        }
    )

    with _mock_ws_thread("localhost", unused_tcp_port, []):
        thread = ErtThread(target=main, args=[["script.py", str(tmp_path)]])
        thread.start()
        _wait_until(
            lambda: f"Could not find file {JOBS_FILE}, retrying" in caplog.text,
            2,
            "Did not get expected log message from missing jobs.json",
        )
        (tmp_path / JOBS_FILE).write_text(jobs_json)
        lock.release()
        thread.join()


@pytest.mark.parametrize(
    "is_interactive_run, ens_id",
    [(False, None), (False, "1234"), (True, None), (True, "1234")],
)
def test_setup_reporters(is_interactive_run, ens_id):
    reporters = _setup_reporters(is_interactive_run, ens_id, "")

    if not is_interactive_run and not ens_id:
        assert len(reporters) == 1
        assert not any(isinstance(r, Event) for r in reporters)

    if not is_interactive_run and ens_id:
        assert len(reporters) == 2
        assert any(isinstance(r, Event) for r in reporters)

    if is_interactive_run and ens_id:
        assert len(reporters) == 1
        assert any(isinstance(r, Interactive) for r in reporters)


@pytest.mark.usefixtures("use_tmpdir")
def test_job_dispatch_kills_itself_after_unsuccessful_job(unused_tcp_port):
    host = "localhost"
    port = unused_tcp_port
    jobs_json = json.dumps({"ens_id": "_id_", "dispatch_url": f"ws://localhost:{port}"})

    with patch("_ert.forward_model_runner.cli.os.killpg") as mock_killpg, patch(
        "_ert.forward_model_runner.cli.os.getpgid"
    ) as mock_getpgid, patch(
        "_ert.forward_model_runner.cli.open", new=mock_open(read_data=jobs_json)
    ), patch("_ert.forward_model_runner.cli.ForwardModelRunner") as mock_runner:
        mock_runner.return_value.run.return_value = [
            Init([], 0, 0),
            Finish().with_error("overall bad run"),
        ]
        mock_getpgid.return_value = 17

        with _mock_ws_thread(host, port, []):
            main(["script.py"])

        mock_killpg.assert_called_with(17, signal.SIGKILL)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No oom_score on MacOS")
def test_killed_by_oom(tmp_path, monkeypatch):
    """Test out-of-memory detection for pid and descendants based
    on a mocked dmesg system utility."""
    parent_pid = 666
    child_pid = 667
    killed_pid = child_pid

    dmesg_path = tmp_path / "dmesg"
    dmesg_path.write_text(
        f"#!/bin/sh\necho 'Out of memory: Killed process {killed_pid}'; exit 0"
    )
    dmesg_path.chmod(dmesg_path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    assert killed_by_oom({parent_pid, child_pid})
    assert killed_by_oom({child_pid})
    assert not killed_by_oom({parent_pid})
    assert not killed_by_oom({child_pid + 1})
