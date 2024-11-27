import asyncio

import pytest

from _ert.forward_model_runner.reporting.message import Exited, Finish, Start
from _ert.forward_model_runner.runner import ForwardModelRunner


@pytest.mark.usefixtures("use_tmpdir")
async def test_run_one_fm_step_with_an_integer_arg_is_actually_a_fractional():
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
    event_queue = asyncio.Queue()
    runner = ForwardModelRunner(data, event_queue)
    await runner.run([])
    start_msg_count = 0
    while not event_queue.empty():
        event = await event_queue.get()
        if isinstance(event, Start):
            start_msg_count += 1
            assert not event.success(), "fm_step should not start with success"

    assert start_msg_count == 1, "There should be 1 start message"


@pytest.mark.usefixtures("use_tmpdir")
async def test_run_given_one_fm_step_with_missing_file_and_one_file_present():
    with open("a_file", "w", encoding="utf-8") as f:  # noqa: ASYNC230
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
    event_queue = asyncio.Queue()
    runner = ForwardModelRunner(data, event_queue)
    await runner.run([])

    statuses = []
    while not event_queue.empty():
        event = await event_queue.get()
        statuses.append(event)

    starts = [e for e in statuses if isinstance(e, Start)]
    assert len(starts) == 2, "There should be 2 start messages"
    assert starts[0].success(), "first fm_step should start with success"
    assert not starts[1].success(), "second fm_step should not start with success"

    exits = [e for e in statuses if isinstance(e, Exited)]
    assert len(exits) == 1, "There should be 1 exit message"
    assert exits[0].success(), "first fm_step should exit with success"

    assert isinstance(statuses[-1], Finish), "last message should be Finish"
    assert not statuses[-1].success(), "Finish status should not be success"
