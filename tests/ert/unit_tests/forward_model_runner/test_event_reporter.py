import os
import time

import pytest

from _ert.events import (
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    event_from_json,
)
from _ert.forward_model_runner.forward_model_step import ForwardModelStep
from _ert.forward_model_runner.reporting import Event
from _ert.forward_model_runner.reporting.message import (
    Exited,
    Finish,
    Init,
    ProcessTreeStatus,
    Running,
    Start,
)
from _ert.forward_model_runner.reporting.statemachine import TransitionError
from tests.ert.utils import MockZMQServer


def _wait_until(condition, timeout, fail_msg):
    start = time.time()
    while not condition():
        assert start + timeout > time.time(), fail_msg
        time.sleep(0.1)


def test_report_with_successful_start_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )
    with MockZMQServer(unused_tcp_port) as mock_server:
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Start(fmstep1))
        reporter.report(Finish())

    assert len(mock_server.messages) == 1
    event = event_from_json(mock_server.messages[0])
    assert type(event) is ForwardModelStepStart
    assert event.ensemble == "ens_id"
    assert event.real == "0"
    assert event.fm_step == "0"
    assert os.path.basename(event.std_out) == "stdout"
    assert os.path.basename(event.std_err) == "stderr"


def test_report_with_failed_start_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)

    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )

    with MockZMQServer(unused_tcp_port) as mock_server:
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))

        msg = Start(fmstep1).with_error("massive_failure")

        reporter.report(msg)
        reporter.report(Finish())

    assert len(mock_server.messages) == 2
    event = event_from_json(mock_server.messages[1])
    assert type(event) is ForwardModelStepFailure
    assert event.error_msg == "massive_failure"


async def test_report_with_successful_exit_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )

    with MockZMQServer(unused_tcp_port) as mock_server:
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Exited(fmstep1, 0))
        reporter.report(Finish().with_error("failed"))

    assert len(mock_server.messages) == 1
    event = event_from_json(mock_server.messages[0])
    assert type(event) is ForwardModelStepSuccess


def test_report_with_failed_exit_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )

    with MockZMQServer(unused_tcp_port) as mock_server:
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Exited(fmstep1, 1).with_error("massive_failure"))
        reporter.report(Finish())

    assert len(mock_server.messages) == 1
    event = event_from_json(mock_server.messages[0])
    assert type(event) is ForwardModelStepFailure
    assert event.error_msg == "massive_failure"


def test_report_with_running_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )

    with MockZMQServer(unused_tcp_port) as mock_server:
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=100, rss=10)))
        reporter.report(Finish())

    assert len(mock_server.messages) == 1
    event = event_from_json(mock_server.messages[0])
    assert type(event) is ForwardModelStepRunning
    assert event.max_memory_usage == 100
    assert event.current_memory_usage == 10


def test_report_only_job_running_for_successful_run(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )

    with MockZMQServer(unused_tcp_port) as mock_server:
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=100, rss=10)))
        reporter.report(Finish())

    assert len(mock_server.messages) == 1


def test_report_with_failed_finish_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )

    with MockZMQServer(unused_tcp_port) as mock_server:
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=100, rss=10)))
        reporter.report(Finish().with_error("massive_failure"))

    assert len(mock_server.messages) == 1


def test_report_inconsistent_events(unused_tcp_port):
    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)

    with (
        pytest.raises(
            TransitionError,
            match=r"Illegal transition None -> \(MessageType<Finish>,\)",
        ),
    ):
        reporter.report(Finish())


def test_report_with_failed_reporter_but_finished_jobs(unused_tcp_port):
    # this is to show when the reporter fails ert won't crash nor
    # staying hanging but instead finishes up the job;
    # see reporter._event_publisher_thread.join()
    # also assert reporter._timeout_timestamp is None
    # meaning Finish event initiated _timeout and timeout was reached
    # which then sets _timeout_timestamp=None

    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    with MockZMQServer(unused_tcp_port) as mock_server:
        reporter = Event(
            evaluator_url=url, ack_timeout=2, max_retries=0, finished_event_timeout=2
        )
        fmstep1 = ForwardModelStep(
            {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
        )

        mock_server.signal(1)  # prevent router to receive messages
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=100, rss=10)))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=1100, rss=10)))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=1100, rss=10)))
        reporter.report(Finish())
        if reporter._event_publisher_thread.is_alive():
            reporter._event_publisher_thread.join()
        assert reporter._done.is_set()
    assert len(mock_server.messages) == 0, "expected 0 Job running messages"


def test_report_with_reconnected_reporter_but_finished_jobs(unused_tcp_port):
    # this is to show when the reporter fails but reconnects
    # reporter still manages to send events and completes fine
    # see assert reporter._timeout_timestamp is not None
    # meaning Finish event initiated _timeout but timeout wasn't reached since
    # it finished succesfully

    host = "localhost"
    url = f"tcp://{host}:{unused_tcp_port}"
    with MockZMQServer(unused_tcp_port) as mock_server:
        reporter = Event(evaluator_url=url, ack_timeout=1, max_retries=1)
        fmstep1 = ForwardModelStep(
            {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
        )

        mock_server.signal(1)  # prevent router to receive messages
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=100, rss=10)))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=1100, rss=10)))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=1100, rss=10)))
        mock_server.signal(0)  # enable router to receive messages
        reporter.report(Finish())
        if reporter._event_publisher_thread.is_alive():
            reporter._event_publisher_thread.join()
        assert reporter._done.is_set()
    assert len(mock_server.messages) == 3, "expected 3 Job running messages"
