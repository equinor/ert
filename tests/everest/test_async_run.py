from functools import partial
from time import sleep

from everest.util import async_run


def workflow_good():
    sleep(0.1)


def workflow_bad():
    sleep(0.1)
    raise RuntimeError("I am a failure")


def _on_runner_error(error, expected_error):
    # Compare repr since we can't compare RuntimeErrors directly
    assert repr(error) == repr(expected_error)


def _on_runner_finished(_, error, expected_error):
    # Compare repr since we can't compare RuntimeErrors directly
    assert repr(error) == repr(expected_error)


def test_async_run():
    # test successful run
    expected_error = None
    async_run(
        workflow_good,
        on_error=partial(_on_runner_error, expected_error=expected_error),
        on_finished=partial(_on_runner_finished, expected_error=expected_error),
    )

    # test failing run
    expected_error = RuntimeError("I am a failure")
    async_run(
        workflow_bad,
        on_error=partial(_on_runner_error, expected_error=expected_error),
        on_finished=partial(_on_runner_finished, expected_error=expected_error),
    )
