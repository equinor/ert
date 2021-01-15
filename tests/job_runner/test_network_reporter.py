import sys
from datetime import datetime as dt
from unittest import TestCase

from job_runner.job import Job
from job_runner.reporting import Network
from job_runner.reporting.message import Exited, Init, Finish

from unittest.mock import patch

import pytest

# Network reporting is disabled temporarily in python/ert_logger/__init__.py
# according to issue #1095.
@pytest.mark.skip(reason="Logging to the network is currently disabled")
class NetworkReporterTests(TestCase):
    def setUp(self):
        self.reporter = Network()

    @patch("job_runner.reporting.network.requests.post")
    def test_init_msg(self, post_mock):
        self.reporter.report(Init([], 0, 1))

        self.assertTrue(post_mock.called)

    @patch("job_runner.reporting.network.requests.post")
    def test_failed_job_is_reported(self, post_mock):
        self.reporter.start_time = dt.now()
        job = Job({"name": "failing job", "executable": "/dev/null", "argList": []}, 0)

        self.reporter.report(Exited(job, 9).with_error("failed"))
        _, data = post_mock.call_args

        self.assertTrue(post_mock.called, "post not called for failed Exit")
        self.assertIn('"status": "exit"', data["data"], "no exit in data")
        self.assertIn('"error": true', data["data"], "no set err flag in data")

    @patch("job_runner.reporting.network.requests.post")
    def test_successful_job_not_reported(self, post_mock):
        self.reporter.report(Exited(None, 9))

        self.assertFalse(post_mock.called, "post called on successful Exit")

    @patch("job_runner.reporting.network.requests.post")
    def test_successful_forward_model_reported(self, post_mock):
        self.reporter.start_time = dt.now()

        self.reporter.report(Finish())
        _, data = post_mock.call_args

        self.assertTrue(post_mock.called, "post not called on OK Finish")
        self.assertIn('"status": "OK"', data["data"], "no OK in data")

    @patch("job_runner.reporting.network.requests.post")
    def test_failed_forward_model_not_reported(self, post_mock):
        self.reporter.report(Finish().with_error("failed"))

        self.assertFalse(post_mock.called, "post called on failed Finish")
