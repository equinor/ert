import os

from job_runner.job import Job
from job_runner.reporting import Event
from job_runner.reporting.event import (
    _FM_JOB_FAILURE,
    _FM_JOB_RUNNING,
    _FM_JOB_START,
    _FM_JOB_SUCCESS,
    _FM_STEP_FAILURE,
    _FM_STEP_START,
    _FM_STEP_SUCCESS,
)
from job_runner.reporting.message import Exited, Finish, Init, Running, Start


def test_report_with_init_message_argument(tmpdir):

    reporter = Event(event_log=tmpdir / "event_log")
    job1 = Job({"name": "job1", "stdout": "/stdout", "stderr": "/stderr"}, 0)

    reporter.report(Init([job1], 1, 19, ee_id="ee_id", real_id=0, stage_id=0))

    with open(reporter._event_log, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        assert '"name": "job1"' in lines[0], "log missing job1"
        assert f'"type": "{_FM_STEP_START}"' in lines[0], "logged wrong type"
        assert (
            '"source": "/ert/ee/ee_id/real/0/stage/0/step/0"' in lines[0]
        ), "bad source"


def test_report_with_successful_start_message_argument(tmpdir):
    reporter = Event(event_log=tmpdir / "event_log")
    job1 = Job({"name": "job1", "stdout": "/stdout", "stderr": "/stderr"}, 0)
    reporter.report(Init([job1], 1, 19, ee_id="ee_id", real_id=0, stage_id=0))
    msg = Start(job1)

    reporter.report(msg)

    with open(reporter._event_log, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert f'"type": "{_FM_JOB_START}"' in lines[1], "logged wrong type"
        assert (
            '"source": "/ert/ee/ee_id/real/0/stage/0/step/0/job/0"' in lines[1]
        ), "bad source"


def test_report_with_failed_start_message_argument(tmpdir):
    reporter = Event(event_log=tmpdir / "event_log")

    job1 = Job({"name": "job1"}, 0)
    reporter.report(Init([job1], 1, 19, ee_id="ee_id", real_id=0, stage_id=0))

    msg = Start(job1).with_error("massive_failure")

    reporter.report(msg)

    with open(reporter._event_log, "r") as f:
        lines = f.readlines()
        assert len(lines) == 3
        assert f'"type": "{_FM_JOB_FAILURE}"' in lines[2], "logged wrong type"
        assert '"error_msg": "massive_failure"' in lines[2], "log missing error message"


def test_report_with_successful_exit_message_argument(tmpdir):
    reporter = Event(event_log=tmpdir / "event_log")
    job1 = Job({"name": "job1"}, 0)
    reporter.report(Init([job1], 1, 19, ee_id="ee_id", real_id=0, stage_id=0))
    reporter.report(Exited(job1, 0))

    with open(reporter._event_log, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert f'"type": "{_FM_JOB_SUCCESS}"' in lines[1], "logged wrong type"


def test_report_with_failed_exit_message_argument(tmpdir):
    reporter = Event(event_log=tmpdir / "event_log")
    job1 = Job({"name": "job1"}, 0)
    reporter.report(Init([job1], 1, 19, ee_id="ee_id", real_id=0, stage_id=0))
    reporter.report(Exited(job1, 1).with_error("massive_failure"))

    with open(reporter._event_log, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert '"error_msg": "massive_failure"' in lines[1], "log missing error message"
        assert f'"type": "{_FM_JOB_FAILURE}"' in lines[1], "logged wrong type"


def test_report_with_running_message_argument(tmpdir):
    reporter = Event(event_log=tmpdir / "event_log")
    job1 = Job({"name": "job1"}, 0)

    reporter.report(Init([job1], 1, 19, ee_id="ee_id", real_id=0, stage_id=0))
    reporter.report(Running(job1, 100, 10))

    with open(reporter._event_log, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert f'"type": "{_FM_JOB_RUNNING}"' in lines[1], "logged wrong type"
        assert '"max_memory_usage": 100' in lines[1], "log missing max_memory_usage"
        assert (
            '"current_memory_usage": 10' in lines[1]
        ), "log missing current_memory_usage"


def test_report_with_successful_finish_message_argument(tmpdir):
    reporter = Event(event_log=tmpdir / "event_log")
    job1 = Job({"name": "job1"}, 0)

    reporter.report(Init([job1], 1, 19, ee_id="ee_id", real_id=0, stage_id=0))
    reporter.report(Running(job1, 100, 10))
    reporter.report(Finish())

    with open(reporter._event_log, "r") as f:
        lines = f.readlines()
        assert len(lines) == 3
        assert f'"type": "{_FM_STEP_SUCCESS}"' in lines[2], "logged wrong type"


def test_report_with_failed_finish_message_argument(tmpdir):
    reporter = Event(event_log=tmpdir / "event_log")
    job1 = Job({"name": "job1"}, 0)

    reporter.report(Init([job1], 1, 19, ee_id="ee_id", real_id=0, stage_id=0))
    reporter.report(Running(job1, 100, 10))
    reporter.report(Finish().with_error("massive_failure"))

    with open(reporter._event_log, "r") as f:
        lines = f.readlines()
        assert len(lines) == 3
        assert f'"type": "{_FM_STEP_FAILURE}"' in lines[2], "logged wrong type"
        assert '"error_msg": "massive_failure"' in lines[2], "log missing error message"


def test_report_startup_clearing_of_event_log_file(tmpdir):
    reporter1 = Event(event_log=tmpdir / "event_log")
    job1 = Job({"name": "job1"}, 0)
    reporter1.report(Init([job1], 1, 19, ee_id="ee_id", real_id=0, stage_id=0))

    reporter2 = Event(event_log=tmpdir / "event_log")

    assert os.path.getsize(tmpdir / "event_log") == 0
