import importlib
import json
import os
import signal
import stat
import sys
from subprocess import Popen
from textwrap import dedent
from unittest.mock import mock_open, patch

import psutil
import pytest

from _ert_job_runner.cli import _setup_reporters, main
from _ert_job_runner.reporting import Event, Interactive
from _ert_job_runner.reporting.message import Finish, Init
from tests.utils import _mock_ws_thread, wait_until


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
                "max_running": None,
                "min_arg": 1,
                "arg_types": [],
                "max_arg": None,
            }
        ],
        "run_id": "",
        "ert_pid": "",
    }

    with open("jobs.json", "w", encoding="utf-8") as f:
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
        "_ert_job_runner.job_dispatch"
    ).origin
    # pylint: disable=consider-using-with
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
def test_job_dispatch_run_subset_specified_as_parmeter():
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
                "max_running": None,
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
                "max_running": None,
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
                "max_running": None,
                "min_arg": 1,
                "arg_types": [],
                "max_arg": None,
            },
        ],
        "run_id": "",
        "ert_pid": "",
    }

    with open("jobs.json", "w", encoding="utf-8") as f:
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
        "_ert_job_runner.job_dispatch"
    ).origin
    # pylint: disable=consider-using-with
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


@pytest.mark.usefixtures("use_tmpdir")
def test_no_jobs_json_file():
    with pytest.raises(IOError):
        main(["script.py", os.path.realpath(os.curdir)])


@pytest.mark.usefixtures("use_tmpdir")
def test_no_json_jobs_json_file():
    path = os.path.realpath(os.curdir)
    jobs_file = os.path.join(path, "jobs.json")

    with open(jobs_file, "w", encoding="utf-8") as f:
        f.write("not json")

    with pytest.raises(OSError):
        main(["script.py", path])


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

    with patch("_ert_job_runner.cli.os.killpg") as mock_killpg, patch(
        "_ert_job_runner.cli.os.getpgid"
    ) as mock_getpgid, patch(
        "_ert_job_runner.cli.open", new=mock_open(read_data=jobs_json)
    ), patch(
        "_ert_job_runner.cli.JobRunner"
    ) as mock_runner:
        mock_runner.return_value.run.return_value = [
            Init([], 0, 0),
            Finish().with_error("overall bad run"),
        ]
        mock_getpgid.return_value = 17

        with _mock_ws_thread(host, port, []):
            main(["script.py"])

        mock_killpg.assert_called_with(17, signal.SIGKILL)
