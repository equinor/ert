import os

import pytest

from _ert_com_protocol import (
    JOB_FAILURE,
    JOB_RUNNING,
    JOB_START,
    JOB_SUCCESS,
    DispatcherMessage,
)
from _ert_job_runner.job import Job
from _ert_job_runner.reporting import Protobuf
from _ert_job_runner.reporting.message import Exited, Finish, Init, Running, Start
from _ert_job_runner.reporting.statemachine import TransitionError
from tests.utils import _mock_ws_thread


def test_report_with_successful_start_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Protobuf(experiment_url=url)
    job1 = Job({"name": "job1", "stdout": "stdout", "stderr": "stderr"}, 0)
    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(
            Init(
                [job1],
                1,
                19,
                experiment_id="experiment_id",
                ens_id="ens_id",
                real_id=0,
                step_id=0,
            )
        )
        reporter.report(Start(job1))
        reporter.report(Finish())

    assert len(lines) == 1
    event = DispatcherMessage()
    event.ParseFromString(lines[0])
    assert event.WhichOneof("object") == "job"
    # pylint: disable=no-member
    assert event.job.status == JOB_START
    assert event.job.id.index == 0
    assert event.job.id.step.step == 0
    assert event.job.id.step.realization.realization == 0
    assert event.job.id.step.realization.ensemble.id == "ens_id"
    assert event.job.id.step.realization.ensemble.experiment.id == "experiment_id"
    assert os.path.basename(event.job.stdout) == "stdout"
    assert os.path.basename(event.job.stderr) == "stderr"


def test_report_with_failed_start_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Protobuf(experiment_url=url)

    job1 = Job({"name": "job1", "stdout": "stdout", "stderr": "stderr"}, 0)

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(
            Init(
                [job1],
                1,
                19,
                ens_id="ens_id",
                real_id=0,
                step_id=0,
                experiment_id="experiment_id",
            )
        )

        msg = Start(job1).with_error("massive_failure")

        reporter.report(msg)
        reporter.report(Finish())

    assert len(lines) == 2
    event = DispatcherMessage()
    event.ParseFromString(lines[1])
    # pylint: disable=no-member
    assert event.job.status == JOB_FAILURE
    assert event.job.error == "massive_failure"


def test_report_with_successful_exit_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Protobuf(experiment_url=url)
    job1 = Job({"name": "job1", "stdout": "stdout", "stderr": "stderr"}, 0)

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(
            Init(
                [job1],
                1,
                19,
                ens_id="ens_id",
                real_id=0,
                step_id=0,
                experiment_id="experiment_id",
            )
        )
        reporter.report(Exited(job1, 0))
        reporter.report(Finish().with_error("failed"))

    assert len(lines) == 1
    event = DispatcherMessage()
    event.ParseFromString(lines[0])
    assert event.WhichOneof("object") == "job"
    # pylint: disable=no-member
    assert event.job.status == JOB_SUCCESS


def test_report_with_failed_exit_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Protobuf(experiment_url=url)
    job1 = Job({"name": "job1", "stdout": "stdout", "stderr": "stderr"}, 0)

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(
            Init(
                [job1],
                1,
                19,
                ens_id="ens_id",
                real_id=0,
                step_id=0,
                experiment_id="experiment_id",
            )
        )
        reporter.report(Exited(job1, 1).with_error("massive_failure"))
        reporter.report(Finish())

    assert len(lines) == 1
    event = DispatcherMessage()
    event.ParseFromString(lines[0])
    assert event.WhichOneof("object") == "job"
    # pylint: disable=no-member
    assert event.job.status == JOB_FAILURE
    assert event.job.error == "massive_failure"
    assert event.job.exit_code == 1


def test_report_with_running_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Protobuf(experiment_url=url)
    job1 = Job({"name": "job1", "stdout": "stdout", "stderr": "stderr"}, 0)

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(
            Init(
                [job1],
                1,
                19,
                ens_id="ens_id",
                real_id=0,
                step_id=0,
                experiment_id="experiment_id",
            )
        )
        reporter.report(Running(job1, 100, 10))
        reporter.report(Finish())

    assert len(lines) == 1
    event = DispatcherMessage()
    event.ParseFromString(lines[0])
    assert event.WhichOneof("object") == "job"
    # pylint: disable=no-member
    assert event.job.status == JOB_RUNNING
    assert event.job.max_memory == 100
    assert event.job.current_memory == 10


def test_report_only_job_running_for_successful_run(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Protobuf(experiment_url=url)
    job1 = Job({"name": "job1", "stdout": "stdout", "stderr": "stderr"}, 0)

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(
            Init(
                [job1],
                1,
                19,
                ens_id="ens_id",
                real_id=0,
                step_id=0,
                experiment_id="experiment_id",
            )
        )
        reporter.report(Running(job1, 100, 10))
        reporter.report(Finish())

    assert len(lines) == 1


def test_report_with_failed_finish_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Protobuf(experiment_url=url)
    job1 = Job({"name": "job1", "stdout": "stdout", "stderr": "stderr"}, 0)

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(
            Init(
                [job1],
                1,
                19,
                ens_id="ens_id",
                real_id=0,
                step_id=0,
                experiment_id="experiment_id",
            )
        )
        reporter.report(Running(job1, 100, 10))
        reporter.report(Finish().with_error("massive_failure"))

    assert len(lines) == 1


def test_report_inconsistent_events(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Protobuf(experiment_url=url)

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines), pytest.raises(
        TransitionError,
        match=r"Illegal transition None -> \(MessageType<Finish>,\)",
    ):
        reporter.report(Finish())
