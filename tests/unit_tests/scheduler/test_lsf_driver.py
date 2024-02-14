import os
import stat
from contextlib import ExitStack as does_not_raise
from pathlib import Path

import pytest

from ert.scheduler import LsfDriver


@pytest.fixture
def capturing_bsub(monkeypatch, tmp_path):
    os.chdir(tmp_path)
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bsub_path = bin_path / "bsub"
    bsub_path.write_text(
        "#!/bin/sh\n"
        "echo $@ > captured_bsub_args\n"
        "echo 'Job <1>' is submitted to normal queue",
        encoding="utf-8",
    )
    bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_with_named_queue():
    driver = LsfDriver(queue_name="foo")
    await driver.submit(0, "sleep")
    assert "-q foo" in Path("captured_bsub_args").read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_bsub")
async def test_submit_with_default_queue():
    driver = LsfDriver()
    await driver.submit(0, "sleep")
    assert "-q" not in Path("captured_bsub_args").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "bsub_script, expectation",
    [
        pytest.param(
            "echo 'Job <444> is submitted to default queue'",
            does_not_raise(),
            id="all-good",
        ),
        pytest.param(
            "echo 'Job <444> is subtted to default queue'",
            pytest.raises(RuntimeError, match="subtte"),
            id="typo_from_bsub",
        ),
        pytest.param(
            "echo 'Job <0> is submitted to default queue'",
            does_not_raise(),
            id="zero_job_id",
        ),
        pytest.param(
            "echo 'Job <-1> is submitted to default queue'",
            pytest.raises(RuntimeError, match="Could not understand"),
            id="negative_job_id",
        ),
        pytest.param(
            "echo 'Job <fuzz> is submitted to default queue'",
            pytest.raises(RuntimeError, match="Could not understand"),
            id="non_number_job_id",
        ),
        pytest.param(
            "echo 'Job <1.4> is submitted to default queue'",
            pytest.raises(RuntimeError, match="Could not understand"),
            id="floating_job_id",
        ),
        pytest.param("exit 1", pytest.raises(RuntimeError), id="plain_exit_1"),
        pytest.param(
            "echo no_go >&2; exit 1",
            pytest.raises(RuntimeError, match="no_go"),
            id="exit_with_stderr",
        ),
        pytest.param(
            "exit 0", pytest.raises(RuntimeError), id="exit_0_but_empty_stdout"
        ),
    ],
)
async def test_faulty_bsub(monkeypatch, tmp_path, bsub_script, expectation):
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bsub_path = bin_path / "bsub"
    bsub_path.write_text(f"#!/bin/sh\n{bsub_script}")
    bsub_path.chmod(bsub_path.stat().st_mode | stat.S_IEXEC)
    driver = LsfDriver()
    with expectation:
        await driver.submit(0, "sleep")
