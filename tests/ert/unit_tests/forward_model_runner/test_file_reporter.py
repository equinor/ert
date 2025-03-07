import os
import os.path

import pytest

from _ert.forward_model_runner.forward_model_step import ForwardModelStep
from _ert.forward_model_runner.reporting import File
from _ert.forward_model_runner.reporting.message import (
    Exited,
    Finish,
    Init,
    ProcessTreeStatus,
    Running,
    Start,
)
from ert.constant_filenames import ERROR_file, LOG_file, STATUS_file, STATUS_json


@pytest.fixture
def reporter():
    return File()


@pytest.mark.usefixtures("use_tmpdir")
def test_report_with_init_message_argument(reporter):
    r = reporter
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "/stdout", "stderr": "/stderr"}, 0
    )

    r.report(Init([fmstep1], 1, 19))

    with open(STATUS_file, encoding="utf-8") as f:
        assert "Current host" in f.readline(), "STATUS file missing expected value"
    with open(STATUS_json, encoding="utf-8") as f:
        content = "".join(f.readlines())
        assert '"name": "fmstep1"' in content, "status.json missing fmstep1"
        assert '"status": "Waiting"' in content, "status.json missing Waiting status"


@pytest.mark.usefixtures("use_tmpdir")
def test_report_with_successful_start_message_argument(reporter):
    msg = Start(
        ForwardModelStep(
            {
                "name": "fmstep1",
                "stdout": "/stdout.0",
                "stderr": "/stderr.0",
                "argList": ["--foo", "1", "--bar", "2"],
                "executable": "/bin/sh",
            },
            0,
        )
    )
    reporter.status_dict = reporter._init_step_status_dict(msg.timestamp, 0, [msg.step])

    reporter.report(msg)

    with open(STATUS_file, encoding="utf-8") as f:
        assert "fmstep1" in f.readline(), "STATUS file missing fmstep1"
    with open(LOG_file, encoding="utf-8") as f:
        assert "Calling: /bin/sh --foo 1 --bar 2" in f.readline(), (
            """JOB_LOG file missing executable and arguments"""
        )

    with open(STATUS_json, encoding="utf-8") as f:
        content = "".join(f.readlines())
        assert '"status": "Running"' in content, "status.json missing Running status"
        assert '"start_time": null' not in content, "start_time not set"


@pytest.mark.usefixtures("use_tmpdir")
def test_report_with_failed_start_message_argument(reporter):
    msg = Start(ForwardModelStep({"name": "fmstep1"}, 0)).with_error("massive_failure")
    reporter.status_dict = reporter._init_step_status_dict(msg.timestamp, 0, [msg.step])

    reporter.report(msg)

    with open(STATUS_file, encoding="utf-8") as f:
        assert "EXIT: -10/massive_failure" in f.readline(), (
            "STATUS file missing EXIT message"
        )
    with open(STATUS_json, encoding="utf-8") as f:
        content = "".join(f.readlines())
        assert '"status": "Failure"' in content, "status.json missing Failure status"
        assert '"error": "massive_failure"' in content, (
            "status.json missing error message"
        )
    assert reporter.status_dict["steps"][0]["end_time"] is not None, (
        "end_time not set for fmstep1"
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_report_with_successful_exit_message_argument(reporter):
    msg = Exited(ForwardModelStep({"name": "fmstep1"}, 0), 0)
    reporter.status_dict = reporter._init_step_status_dict(msg.timestamp, 0, [msg.step])

    reporter.report(msg)

    with open(STATUS_json, encoding="utf-8") as f:
        content = "".join(f.readlines())
        assert '"status": "Success"' in content, "status.json missing Success status"


@pytest.mark.usefixtures("use_tmpdir")
def test_report_with_failed_exit_message_argument(reporter):
    msg = Exited(ForwardModelStep({"name": "fmstep1"}, 0), 1).with_error(
        "massive_failure"
    )
    reporter.status_dict = reporter._init_step_status_dict(msg.timestamp, 0, [msg.step])

    reporter.report(msg)

    with open(STATUS_file, encoding="utf-8") as f:
        assert "EXIT: 1/massive_failure" in f.readline()
    with open(ERROR_file, encoding="utf-8") as f:
        content = "".join(f.readlines())
        assert "<step>fmstep1</step>" in content, "ERROR file missing fmstep"
        assert "<reason>massive_failure</reason>" in content, (
            "ERROR file missing reason"
        )
        assert "stderr: Not redirected" in content, (
            "ERROR had invalid stderr information"
        )
    with open(STATUS_json, encoding="utf-8") as f:
        content = "".join(f.readlines())
        assert '"status": "Failure"' in content, "status.json missing Failure status"
        assert '"error": "massive_failure"' in content, (
            "status.json missing error message"
        )
    assert reporter.status_dict["steps"][0]["end_time"] is not None


@pytest.mark.usefixtures("use_tmpdir")
def test_report_with_running_message_argument(reporter):
    msg = Running(
        ForwardModelStep({"name": "fmstep1"}, 0),
        ProcessTreeStatus(max_rss=100, rss=10, cpu_seconds=1.1),
    )
    reporter.status_dict = reporter._init_step_status_dict(msg.timestamp, 0, [msg.step])

    reporter.report(msg)

    with open(STATUS_json, encoding="utf-8") as f:
        content = "".join(f.readlines())
        assert '"status": "Running"' in content, "status.json missing status"
        assert '"max_memory_usage": 100' in content, (
            "status.json missing max_memory_usage"
        )
        assert '"current_memory_usage": 10' in content, (
            "status.json missing current_memory_usage"
        )
        assert '"cpu_seconds": 1.1' in content, "status.json missing cpu_seconds"


@pytest.mark.usefixtures("use_tmpdir")
def test_report_with_successful_finish_message_argument(reporter):
    msg = Finish()
    reporter.status_dict = reporter._init_step_status_dict(msg.timestamp, 0, [])

    reporter.report(msg)


@pytest.mark.usefixtures("use_tmpdir")
def test_dump_error_file_with_stderr(reporter):
    """
    Assert that, in the case of an stderr file, it is included in the XML
    that constitutes the ERROR file.
    The report system is left out, since this was tested in the fail case.
    """
    with open("stderr.out.0", "a", encoding="utf-8") as stderr:
        stderr.write("Error:\n")
        stderr.write("E_MASSIVE_FAILURE\n")

    reporter._dump_error_file(
        ForwardModelStep({"name": "fmstep1", "stderr": "stderr.out.0"}, 0),
        "massive_failure",
    )

    with open(ERROR_file, encoding="utf-8") as f:
        content = "".join(f.readlines())
        assert "E_MASSIVE_FAILURE" in content, "ERROR file missing stderr content"
        assert "<stderr_file>" in content, "ERROR missing stderr_file part"


@pytest.mark.usefixtures("use_tmpdir")
def test_old_file_deletion(reporter):
    # touch all files that are to be removed
    for f in [ERROR_file, STATUS_file]:
        with open(f, "a", encoding="utf-8"):
            pass

    reporter._delete_old_status_files()

    for f in [ERROR_file, STATUS_file]:
        assert not os.path.isfile(f), f"{reporter} was not deleted"


@pytest.mark.usefixtures("use_tmpdir")
def test_status_file_is_correct(reporter):
    """The STATUS file is a file to which we append data about steps as they
    are run. So this involves multiple reports, and should be tested as
    such.
    See https://github.com/equinor/libres/issues/764
    """
    j_1 = ForwardModelStep({"name": "j_1", "executable": "", "argList": []}, 0)
    j_2 = ForwardModelStep({"name": "j_2", "executable": "", "argList": []}, 0)
    init = Init([j_1, j_2], 1, 1)
    start_j_1 = Start(j_1)
    exited_j_1 = Exited(j_1, 0)
    start_j_2 = Start(j_2)
    exited_j_2 = Exited(j_2, 9).with_error("failed horribly")

    for msg in [init, start_j_1, exited_j_1, start_j_2, exited_j_2]:
        reporter.report(msg)

    expected_j1_line = (
        f"{j_1.name():32}: {start_j_1.timestamp:%H:%M:%S} .... "
        f"{exited_j_1.timestamp:%H:%M:%S}  \n"
    )
    expected_j2_line = (
        f"{j_2.name():32}: "
        f"{start_j_2.timestamp:%H:%M:%S} .... "
        f"{exited_j_2.timestamp:%H:%M:%S}   "
        f"EXIT: {exited_j_2.exit_code}/{exited_j_2.error_message}\n"
    )

    with open(STATUS_file, encoding="utf-8") as f:
        for expected in [
            "Current host",
            expected_j1_line,
            expected_j2_line,
        ]:
            assert expected in f.readline()

        # EOF
        assert not f.readline()
