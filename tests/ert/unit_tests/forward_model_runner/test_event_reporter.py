import os
import sys
import time
from unittest.mock import patch

import pytest

from _ert.events import (
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    event_from_json,
)
from _ert.forward_model_runner.client import (
    ClientConnectionClosedOK,
    ClientConnectionError,
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
from tests.ert.utils import _mock_ws_thread


def _wait_until(condition, timeout, fail_msg):
    start = time.time()
    while not condition():
        assert start + timeout > time.time(), fail_msg
        time.sleep(0.1)


def test_report_with_successful_start_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )
    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Start(fmstep1))
        reporter.report(Finish())

    assert len(lines) == 1
    event = event_from_json(lines[0])
    assert type(event) is ForwardModelStepStart
    assert event.ensemble == "ens_id"
    assert event.real == "0"
    assert event.fm_step == "0"
    assert os.path.basename(event.std_out) == "stdout"
    assert os.path.basename(event.std_err) == "stderr"


def test_report_with_failed_start_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)

    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))

        msg = Start(fmstep1).with_error("massive_failure")

        reporter.report(msg)
        reporter.report(Finish())

    assert len(lines) == 2
    event = event_from_json(lines[1])
    assert type(event) is ForwardModelStepFailure
    assert event.error_msg == "massive_failure"


def test_report_with_successful_exit_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Exited(fmstep1, 0))
        reporter.report(Finish().with_error("failed"))

    assert len(lines) == 1
    event = event_from_json(lines[0])
    assert type(event) is ForwardModelStepSuccess


def test_report_with_failed_exit_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Exited(fmstep1, 1).with_error("massive_failure"))
        reporter.report(Finish())

    assert len(lines) == 1
    event = event_from_json(lines[0])
    assert type(event) is ForwardModelStepFailure
    assert event.error_msg == "massive_failure"


def test_report_with_running_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=100, rss=10)))
        reporter.report(Finish())

    assert len(lines) == 1
    event = event_from_json(lines[0])
    assert type(event) is ForwardModelStepRunning
    assert event.max_memory_usage == 100
    assert event.current_memory_usage == 10


def test_report_only_job_running_for_successful_run(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=100, rss=10)))
        reporter.report(Finish())

    assert len(lines) == 1


def test_report_with_failed_finish_message_argument(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )

    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=100, rss=10)))
        reporter.report(Finish().with_error("massive_failure"))

    assert len(lines) == 1


def test_report_inconsistent_events(unused_tcp_port):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)

    lines = []
    with (
        _mock_ws_thread(host, unused_tcp_port, lines),
        pytest.raises(
            TransitionError,
            match=r"Illegal transition None -> \(MessageType<Finish>,\)",
        ),
    ):
        reporter.report(Finish())


@pytest.mark.integration_test
def test_report_with_failed_reporter_but_finished_jobs(unused_tcp_port):
    # this is to show when the reporter fails ert won't crash nor
    # staying hanging but instead finishes up the job;
    # see reporter._event_publisher_thread.join()
    # also assert reporter._timeout_timestamp is None
    # meaning Finish event initiated _timeout and timeout was reached
    # which then sets _timeout_timestamp=None
    mock_send_retry_time = 2

    def mock_send(msg):
        time.sleep(mock_send_retry_time)
        raise ClientConnectionError("Sending failed!")

    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    reporter._reporter_timeout = 4
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )
    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        with patch(
            "_ert.forward_model_runner.client.Client.send", lambda x, y: mock_send(y)
        ):
            reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
            reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=100, rss=10)))
            reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=1100, rss=10)))
            reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=1100, rss=10)))
            # set _stop_timestamp
            reporter.report(Finish())
        if reporter._event_publisher_thread.is_alive():
            reporter._event_publisher_thread.join()
        # set _stop_timestamp to None only when timer stopped
        assert reporter._timeout_timestamp is None
    assert len(lines) == 0, "expected 0 Job running messages"


@pytest.mark.integration_test
@pytest.mark.flaky(reruns=5)
@pytest.mark.skipif(
    sys.platform.startswith("darwin"), reason="Performance can be flaky"
)
def test_report_with_reconnected_reporter_but_finished_jobs(unused_tcp_port):
    # this is to show when the reporter fails but reconnects
    # reporter still manages to send events and completes fine
    # see assert reporter._timeout_timestamp is not None
    # meaning Finish event initiated _timeout but timeout wasn't reached since
    # it finished succesfully
    mock_send_retry_time = 0.1

    def send_func(msg):
        time.sleep(mock_send_retry_time)
        raise ClientConnectionError("Sending failed!")

    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )
    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        with patch("_ert.forward_model_runner.client.Client.send") as patched_send:
            patched_send.side_effect = send_func

            reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
            reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=100, rss=10)))
            reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=200, rss=10)))
            reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=300, rss=10)))

            _wait_until(
                condition=lambda: patched_send.call_count == 3,
                timeout=10,
                fail_msg="10 seconds should be sufficient to send three events",
            )

        # reconnect and continue sending events
        # set _stop_timestamp
        reporter.report(Finish())
        if reporter._event_publisher_thread.is_alive():
            reporter._event_publisher_thread.join()
        # set _stop_timestamp was not set to None since the reporter finished on time
        assert reporter._timeout_timestamp is not None
    assert len(lines) == 3, "expected 3 Job running messages"


@pytest.mark.integration_test
def test_report_with_closed_received_exiting_gracefully(unused_tcp_port):
    # Whenever the receiver end closes the connection, a ConnectionClosedOK is raised
    # The reporter should exit the publisher thread gracefully and not send any
    # more events
    mock_send_retry_time = 3

    def mock_send(msg):
        time.sleep(mock_send_retry_time)
        raise ClientConnectionClosedOK("Connection Closed")

    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    reporter = Event(evaluator_url=url)
    fmstep1 = ForwardModelStep(
        {"name": "fmstep1", "stdout": "stdout", "stderr": "stderr"}, 0
    )
    lines = []
    with _mock_ws_thread(host, unused_tcp_port, lines):
        reporter.report(Init([fmstep1], 1, 19, ens_id="ens_id", real_id=0))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=100, rss=10)))
        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=200, rss=10)))

        # sleep until both Running events have been received
        _wait_until(
            condition=lambda: len(lines) == 2,
            timeout=10,
            fail_msg="Should not take 10 seconds to send two events",
        )

        with patch(
            "_ert.forward_model_runner.client.Client.send", lambda x, y: mock_send(y)
        ):
            reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=300, rss=10)))
            # Make sure the publisher thread exits because it got
            # ClientConnectionClosedOK. If it hangs it could indicate that the
            # exception is not caught/handled correctly
            if reporter._event_publisher_thread.is_alive():
                reporter._event_publisher_thread.join()

        reporter.report(Running(fmstep1, ProcessTreeStatus(max_rss=400, rss=10)))
        reporter.report(Finish())

    # set _stop_timestamp was not set to None since the reporter finished on time
    assert reporter._timeout_timestamp is not None

    # The Running(fmstep1, 300, 10) is popped from the queue, but never sent.
    # The following Running is added to queue along with the sentinel
    assert reporter._event_queue.qsize() == 2
    # None of the messages after ClientConnectionClosedOK was raised, has been sent
    assert len(lines) == 2, "expected 2 Job running messages"
