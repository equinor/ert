import sys
from datetime import datetime as dt
from unittest import TestCase


from job_runner.reporting import Network
from job_runner.reporting.message import Exited, Init

if sys.version_info >= (3, 3):
    from unittest.mock import patch
else:
    from mock import patch


class NetworkReporterTests(TestCase):
    def setUp(self):
        self.reporter = Network()

    @patch("job_runner.reporting.network.requests.post")
    def test_init_msg(self, post_mock):
        self.reporter.report(Init([], 0, 1))

        self.assertTrue(post_mock.called)

    @patch("job_runner.reporting.network.requests.post")
    def test_exited_failure_msg(self, post_mock):
        self.reporter.start_time = dt.now()

        self.reporter.report(Exited(None, 9))

        self.assertTrue(post_mock.called)

    @patch("job_runner.reporting.network.requests.post")
    def test_exited_success_msg(self, post_mock):
        self.reporter.start_time = dt.now()

        self.reporter.report(Exited(None, 9))
        _, data = post_mock.call_args

        self.assertTrue(post_mock.called)
        self.assertIn('"status": "OK"', data["data"])
