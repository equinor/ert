import pytest

from ert.ensemble_evaluator.narratives import dispatch_failing_job


@pytest.mark.consumer_driven_contract_test
def test_dispatchers_with_failing_job(unused_tcp_port):
    with dispatch_failing_job().on_uri(f"ws://localhost:{unused_tcp_port}/dispatch"):
        pass
