import pytest

from _ert.forward_model_runner.reporting.message import Exited, Finish, Start
from _ert.forward_model_runner.runner import ForwardModelRunner


@pytest.mark.usefixtures("use_tmpdir")
def test_run_one_fm_step_with_an_integer_arg_is_actually_a_fractional():
    fm_step_1 = {
        "name": "FM_STEP_1",
        "executable": "echo",
        "stdout": "outfile.stdout.1",
        "stderr": None,
        "argList": ["a_file", "5.12"],
        "min_arg": 1,
        "max_arg": 2,
        "arg_types": ["STRING", "RUNTIME_INT"],
    }

    data = {"jobList": [fm_step_1]}

    runner = ForwardModelRunner(data)
    statuses = list(runner.run([]))
    starts = [e for e in statuses if isinstance(e, Start)]

    assert len(starts) == 1, "There should be 1 start message"
    assert not starts[0].success(), "fm_step should not start with success"


@pytest.mark.usefixtures("use_tmpdir")
def test_run_given_one_fm_step_with_missing_file_and_one_file_present():
    with open("a_file", "w", encoding="utf-8") as f:
        f.write("Hello")

    executable = "echo"

    fm_step_1 = {
        "name": "FM_STEP_0",
        "executable": executable,
        "stdout": "outfile.stdout.0",
        "stderr": None,
        "argList": ["some_file"],
        "min_arg": 1,
        "max_arg": 1,
        "arg_types": ["RUNTIME_FILE"],
    }

    fm_step_0 = {
        "name": "FM_STEP_1",
        "executable": executable,
        "stdout": "outfile.stdout.1",
        "stderr": None,
        "argList": ["5", "a_file"],
        "min_arg": 1,
        "max_arg": 2,
        "arg_types": ["RUNTIME_INT", "RUNTIME_FILE"],
    }

    data = {
        "jobList": [fm_step_0, fm_step_1],
    }

    runner = ForwardModelRunner(data)

    statuses = list(runner.run([]))

    starts = [e for e in statuses if isinstance(e, Start)]
    assert len(starts) == 2, "There should be 2 start messages"
    assert starts[0].success(), "first fm_step should start with success"
    assert not starts[1].success(), "second fm_step should not start with success"

    exits = [e for e in statuses if isinstance(e, Exited)]
    assert len(exits) == 1, "There should be 1 exit message"
    assert exits[0].success(), "first fm_step should exit with success"

    assert isinstance(statuses[-1], Finish), "last message should be Finish"
    assert not statuses[-1].success(), "Finish status should not be success"
