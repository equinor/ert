from __future__ import annotations

import asyncio
import glob
import json
import os
import signal
import stat
import sys
from pathlib import Path
from subprocess import Popen
from textwrap import dedent
from threading import Lock
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import psutil
import pytest

import _ert.forward_model_runner.fm_dispatch
from _ert.events import dispatcher_event_from_json
from _ert.forward_model_runner.fm_dispatch import (
    FORWARD_MODEL_DESCRIPTION_FILE,
    FORWARD_MODEL_TERMINATED_MSG,
    _report_all_messages,
    _setup_reporters,
    fm_dispatch,
)
from _ert.forward_model_runner.forward_model_step import killed_by_oom
from _ert.forward_model_runner.reporting import Event, Interactive, Reporter
from _ert.forward_model_runner.reporting.message import Finish, Init, Message
from _ert.threading import ErtThread
from tests.ert.utils import MockZMQServer, wait_until


@pytest.fixture
def use_custom_setsid(use_tmpdir):
    """macOS doesn't provide /usr/bin/setsid, so we roll our own"""
    Path("setsid").write_text(
        dedent(
            """\
            #!/usr/bin/env python
            import os
            import sys
            os.setsid()
            os.execvp(sys.argv[1], sys.argv[1:])
            """
        ),
        encoding="utf-8",
    )
    os.chmod("setsid", 0o755)


def write_executable(
    directory: Path, contents: str, basename: Path | str = "dummy_executable"
) -> str:
    file_path = directory / basename
    file_path.write_text(contents, encoding="utf-8")
    os.chmod(file_path, stat.S_IRWXU | stat.S_IRWXO | stat.S_IRWXG)
    return str(file_path.resolve())


def write_forward_model_description(contents: dict[str, Any]) -> None:
    Path(FORWARD_MODEL_DESCRIPTION_FILE).write_text(
        json.dumps(contents), encoding="utf-8"
    )


def job_dict(
    executable: str, arglist: list[Any], name: str = "dummy_executable"
) -> dict[str, Any]:
    return {
        "name": name,
        "executable": executable,
        "target_file": None,
        "error_file": None,
        "start_file": None,
        "stdout": "dummy.stdout",
        "stderr": "dummy.stderr",
        "stdin": None,
        "argList": arglist,
        "environment": None,
        "license_path": None,
        "max_running_minutes": None,
        "min_arg": 1,
        "arg_types": [],
        "max_arg": None,
    }


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_custom_setsid")
def test_terminate_steps(tmp_path: Path):
    # Executes itself recursively and sleeps for 100 seconds
    executable = write_executable(
        tmp_path,
        dedent(
            """\
                #!/usr/bin/env python
                import sys, os, time
                counter = eval(sys.argv[1])
                if counter > 0:
                    os.fork()
                    os.execv(sys.argv[0],[sys.argv[0], str(counter - 1) ])
                else:
                    time.sleep(100)""",
        ),
    )

    write_forward_model_description(
        {
            "global_environment": {},
            "global_update_path": {},
            "jobList": [job_dict(executable, ["3"])],
            "run_id": "",
            "ert_pid": "",
        }
    )

    # (we wait for the process below)
    fm_dispatch_process = Popen(
        [os.getcwd() + "/setsid", "fm_dispatch.py", os.getcwd()]
    )

    p = psutil.Process(fm_dispatch_process.pid)

    # Three levels of processes should spawn 8 children in total
    wait_until(lambda: len(p.children(recursive=True)) == 8)

    p.terminate()

    wait_until(lambda: len(p.children(recursive=True)) == 0)

    os.wait()  # allow os to clean up zombie processes


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_memory_profile_is_logged_as_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """This tests that a csv is produced and has basic validity.
    It does not try to verify the validity of the logged RSS values."""
    fm_stepname = "do_nothing"
    scriptname = fm_stepname + ".py"
    fm_step_repeats = 3
    executable = write_executable(
        tmp_path,
        """#!/bin/sh
        sleep 0.5
        exit 0
        """,
        basename=scriptname,
    )
    write_forward_model_description(
        {
            "jobList": [
                {
                    "name": fm_stepname,
                    "executable": executable,
                    "argList": [""],
                }
            ]
            * fm_step_repeats,
        }
    )

    monkeypatch.setattr(
        _ert.forward_model_runner.runner.ForwardModelStep, "MEMORY_POLL_PERIOD", 0.1
    )
    fm_dispatch(["fm_dispatch", os.getcwd()])
    csv_files = glob.glob("logs/memory-profile*csv")
    mem_df = pd.read_csv(csv_files[0], parse_dates=True)
    assert mem_df["timestamp"].is_monotonic_increasing
    assert (mem_df["fm_step_id"].unique() == [0, 1, 2]).all()
    assert mem_df["fm_step_name"].unique() == [fm_stepname]
    assert (mem_df["rss"] >= 0).all()  # 0 has been observed


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_custom_setsid")
def test_fm_dispatch_run_subset_specified_as_parameter(tmp_path: Path):
    executable = write_executable(
        tmp_path,
        "#!/usr/bin/env python\n"
        "import sys, os\n"
        'filename = "step_{}.out".format(sys.argv[1])\n'
        'f = open(filename, "w", encoding="utf-8")\n'
        "f.close()\n",
    )

    write_forward_model_description(
        {
            "global_environment": {},
            "global_update_path": {},
            "jobList": [
                job_dict(executable, ["A"], "step_A"),
                job_dict(executable, ["B"], "step_B"),
                job_dict(executable, ["C"], "step_C"),
            ],
            "run_id": "",
            "ert_pid": "",
        }
    )

    # (we wait for the process below)
    fm_dispatch_process = Popen(
        [os.getcwd() + "/setsid", "fm_dispatch.py", os.getcwd(), "step_B", "step_C"]
    )

    fm_dispatch_process.wait()

    assert not os.path.isfile("step_A.out")
    assert os.path.isfile("step_B.out")
    assert os.path.isfile("step_C.out")


def test_no_jobs_json_file_raises_IOError(tmp_path: Path):
    with pytest.raises(IOError, match=r"No such file or directory: 'jobs.json'"):
        fm_dispatch(["script.py", str(tmp_path)])


def test_invalid_jobs_json_raises_OSError(tmp_path: Path):
    (tmp_path / FORWARD_MODEL_DESCRIPTION_FILE).write_text("not json")

    with pytest.raises(OSError, match="fm_dispatch failed to load JSON-file"):
        fm_dispatch(["script.py", str(tmp_path)])


def test_missing_directory_exits(tmp_path: Path):
    with pytest.raises(SystemExit):
        fm_dispatch(["script.py", str(tmp_path / "non_existent")])


def test_retry_of_jobs_json_file_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    lock = Lock()
    lock.acquire()
    monkeypatch.setattr(
        _ert.forward_model_runner.fm_dispatch, "_wait_for_retry", lock.acquire
    )
    with MockZMQServer() as zmq_server:
        jobs_json = json.dumps(
            {
                "ens_id": "_id_",
                "dispatch_url": zmq_server.uri,
                "jobList": [],
            }
        )

        def create_jobs_file_after_lock():
            wait_until(
                lambda: (
                    (f"Could not find file {FORWARD_MODEL_DESCRIPTION_FILE}, retrying")
                    in caplog.text
                ),
                interval=0.1,
                timeout=2,
            )
            (tmp_path / FORWARD_MODEL_DESCRIPTION_FILE).write_text(jobs_json)
            lock.release()

        thread = ErtThread(target=create_jobs_file_after_lock)
        thread.start()
        fm_dispatch(args=["script.py", str(tmp_path)])
        thread.join()


@pytest.mark.parametrize(
    ("is_interactive_run", "ens_id"),
    [(False, None), (False, "1234"), (True, None), (True, "1234")],
)
def test_setup_reporters(is_interactive_run: bool, ens_id: str | None):
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
def test_fm_dispatch_kills_itself_after_unsuccessful_step():
    with (
        patch("_ert.forward_model_runner.fm_dispatch.os.killpg") as mock_killpg,
        patch("_ert.forward_model_runner.fm_dispatch.os.getpgid") as mock_getpgid,
        MockZMQServer() as zmq_server,
        patch(
            "_ert.forward_model_runner.fm_dispatch.ForwardModelRunner"
        ) as mock_runner,
    ):
        mock_runner.return_value.run.return_value = [
            Init([], 0, 0),
            Finish().with_error("overall bad run"),
        ]
        mock_getpgid.return_value = 17
        write_forward_model_description(
            {
                "ens_id": "_id_",
                "dispatch_url": zmq_server.uri,
            }
        )

        fm_dispatch(["script.py"])

        mock_killpg.assert_called_with(17, signal.SIGKILL)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No oom_score on MacOS")
@pytest.mark.usefixtures("use_custom_setsid")
def test_killed_by_oom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_report_all_messages():
    message = MagicMock(spec=Message)
    reporter = MagicMock(spec=Reporter)

    _report_all_messages(iter([message]), [reporter])
    reporter.report.assert_called_once_with(message)


def test_report_all_messages_drops_reporter_on_error():
    message1 = MagicMock(spec=Message)
    message2 = MagicMock(spec=Message)
    reporter = MagicMock(spec=Reporter)

    def raises(*args, **kwargs):
        raise OSError("No space left on device")

    reporter.report.side_effect = raises

    _report_all_messages(iter([message1, message2]), [reporter])
    reporter.report.assert_called_once_with(message1)


@pytest.fixture
def sleep_executable(tmp_path: Path):
    return write_executable(
        tmp_path,
        dedent("""\
               #!/usr/bin/env python
               import time
               time.sleep(180)
               """),
    )


async def wait_for_msg(zmq_server, msg_type: str) -> None:
    while True:
        await asyncio.sleep(0.5)
        if any(
            msg_type in dispatcher_event_from_json(msg).event_type
            for msg in zmq_server.messages
        ):
            return


@pytest.mark.timeout(30)
@pytest.mark.integration_test
@pytest.mark.usefixtures("use_custom_setsid")
async def test_fm_dispatch_sends_exited_event_with_terminated_msg_on_sigterm(
    sleep_executable: str,
):
    async with MockZMQServer() as zmq_server:
        write_forward_model_description(
            {
                "ens_id": "_id_",
                "dispatch_url": zmq_server.uri,
                "jobList": [
                    {
                        "name": "dummy_executable",
                        "executable": sleep_executable,
                        "stdout": "dummy.stdout",
                        "stderr": "dummy.stderr",
                    }
                ],
            }
        )

        fm_dispatch_process = Popen(  # noqa: ASYNC220
            [os.getcwd() + "/setsid", "fm_dispatch.py", os.getcwd()]
        )
        p = psutil.Process(fm_dispatch_process.pid)

        # wait for fm running
        await asyncio.wait_for(
            wait_for_msg(zmq_server, "forward_model_step.start"), timeout=15
        )
        p.terminate()
        # wait for fm_dispatch has been terminated, and sends failure message
        await asyncio.wait_for(
            wait_for_msg(zmq_server, "forward_model_step.failure"), timeout=15
        )
        assert (
            dispatcher_event_from_json(zmq_server.messages[-1]).error_msg
            == FORWARD_MODEL_TERMINATED_MSG
        )


@pytest.mark.timeout(30)
@pytest.mark.integration_test
@pytest.mark.usefixtures("use_custom_setsid")
async def test_fm_dispatch_sends_exited_event_with_terminated_msg_on_terminate_msg(
    sleep_executable: str,
):
    async with MockZMQServer() as zmq_server:
        write_forward_model_description(
            {
                "ens_id": "_id_",
                "dispatch_url": zmq_server.uri,
                "jobList": [
                    {
                        "name": "dummy_executable",
                        "executable": sleep_executable,
                        "stdout": "dummy.stdout",
                        "stderr": "dummy.stderr",
                    }
                ],
            }
        )

        fm_dispatch_process = Popen(  # noqa: ASYNC220
            [os.getcwd() + "/setsid", "fm_dispatch.py", os.getcwd()]
        )

        await asyncio.wait_for(
            wait_for_msg(zmq_server, "forward_model_step.start"), timeout=15
        )
        await zmq_server.send_terminate_message()
        await asyncio.wait_for(
            wait_for_msg(zmq_server, "forward_model_step.failure"), timeout=15
        )
        await zmq_server.no_dealers.wait()
        fm_dispatch_process.wait(timeout=15)
        assert (
            dispatcher_event_from_json(zmq_server.messages[-1]).error_msg
            == FORWARD_MODEL_TERMINATED_MSG
        )


async def test_fm_dispatch_main_signals_sigterm_on_exception(
    capsys: pytest.CaptureFixture[str],
):
    def mock_fm_dispatch_raises(*args):
        raise RuntimeError("forward model critical error")

    with (
        patch("_ert.forward_model_runner.fm_dispatch.os.killpg") as mock_killpg,
        patch("_ert.forward_model_runner.fm_dispatch.os.getpgid") as mock_getpgid,
        patch("_ert.forward_model_runner.fm_dispatch.fm_dispatch") as mock_fm_dispatch,
    ):
        mock_getpgid.return_value = 17
        mock_fm_dispatch.side_effect = mock_fm_dispatch_raises
        _ert.forward_model_runner.fm_dispatch.main()
    assert "forward model critical error" in capsys.readouterr().out

    mock_killpg.assert_called_with(17, signal.SIGTERM)
