import os
import stat
from pathlib import Path
from typing import List

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ert.scheduler import OpenPBSDriver
from ert.scheduler.openpbs_driver import (
    JOBSTATE_INITIAL,
    FinishedEvent,
    StartedEvent,
    _Stat,
)


@given(st.lists(st.sampled_from(["H", "Q", "R", "F"])))
async def test_events_produced_from_jobstate_updates(jobstate_sequence: List[str]):
    # Determine what to expect from the sequence:
    started = False
    finished = False
    if "R" in jobstate_sequence:
        started = True
    if "F" in jobstate_sequence:
        finished = True

    driver = OpenPBSDriver()

    async def mocked_submit(self, iens, *_args, **_kwargs):
        """A mocked submit is speedier than going through a command on disk"""
        self._jobs["1"] = (iens, JOBSTATE_INITIAL)
        self._iens2jobid[iens] = "1"

    driver.submit = mocked_submit.__get__(driver)
    await driver.submit(0, "_")

    for statestr in jobstate_sequence:
        jobstate = _Stat(
            **{"Jobs": {"1": {"job_state": statestr, "Exit_status": 0}}}
        ).jobs["1"]
        await driver._process_job_update("1", jobstate)

    events = []
    while not driver.event_queue.empty():
        events.append(await driver.event_queue.get())

    if started is False and finished is False:
        assert len(events) == 0
        assert driver._jobs["1"] in [(0, "Q"), (0, "H")]
    elif started is True and finished is False:
        assert len(events) == 1
        assert events[0] == StartedEvent(iens=0)
        assert driver._jobs["1"] == (0, "R")
    elif started is True and finished is True:
        assert len(events) <= 2  # The StartedEvent is not required
        assert events[-1] == FinishedEvent(iens=0, returncode=0)
        assert "1" not in driver._jobs


@pytest.fixture
def capturing_qsub(monkeypatch, tmp_path):
    os.chdir(tmp_path)
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    qsub_path = bin_path / "qsub"
    qsub_path.write_text(
        "#!/bin/sh\necho $@ > captured_qsub_args; echo '1'", encoding="utf-8"
    )
    qsub_path.chmod(qsub_path.stat().st_mode | stat.S_IEXEC)


@pytest.mark.usefixtures("capturing_qsub")
async def test_memory_per_job():
    driver = OpenPBSDriver(memory_per_job="10gb")
    await driver.submit(0, "sleep")
    assert " -l mem=10gb " in Path("captured_qsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_qsub")
async def test_no_default_memory_per_job():
    driver = OpenPBSDriver()
    await driver.submit(0, "sleep")
    assert " -l " not in Path("captured_qsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_qsub")
async def test_no_validation_of_memory_per_job():
    # Validation will happen during config parsing
    driver = OpenPBSDriver(memory_per_job="a_lot")
    await driver.submit(0, "sleep")
    assert " -l mem=a_lot " in Path("captured_qsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_qsub")
async def test_job_name():
    driver = OpenPBSDriver()
    await driver.submit(0, "sleep", name="sleepy")
    assert " -Nsleepy " in Path("captured_qsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_qsub")
async def test_job_name_with_prefix():
    driver = OpenPBSDriver(job_prefix="pre_")
    await driver.submit(0, "sleep", name="sleepy")
    assert " -Npre_sleepy " in Path("captured_qsub_args").read_text(encoding="utf-8")
