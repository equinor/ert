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


@pytest.mark.parametrize(
    "mocked_iens2jobid, iens_to_kill, bkill_returncode, bkill_stdout, bkill_stderr, expected_logged_error",
    [
        pytest.param(
            {"1": "11"},
            "1",
            0,
            "Job <11> is being terminated",
            "",
            None,
            id="happy_path",
        ),
        pytest.param(
            {"1": "11"},
            "2",
            1,
            "",
            "",
            "LSF kill failed due to missing",
            id="internal_ert_error",
        ),
        pytest.param(
            {"1": "11"},
            "1",
            255,
            "",
            "Job <22>: No matching job found",
            "No matching job found",
            id="inconsistency_ert_vs_lsf",
        ),
        pytest.param(
            {"1": "11"},
            "1",
            0,
            "wrong_stdout...",
            "",
            "wrong_stdout...",
            id="artifical_bkill_stdout_giving_logged_error",
        ),
        pytest.param(
            {"1": "11"},
            "1",
            0,
            "",
            "wrong_on_stderr",
            "wrong_on_stderr",
            id="artifical_bkill_stderr_giving_logged_error",
        ),
        pytest.param(
            {"1": "11"},
            "1",
            1,
            "",
            "wrong_on_stderr",
            "wrong_on_stderr",
            id="artifical_bkill_stderr_and_returncode_giving_logged_error",
        ),
    ],
)
async def test_kill(
    monkeypatch,
    tmp_path,
    mocked_iens2jobid,
    iens_to_kill,
    bkill_returncode,
    bkill_stdout,
    bkill_stderr,
    expected_logged_error,
    caplog,
):
    os.chdir(tmp_path)
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    bkill_path = bin_path / "bkill"
    bkill_path.write_text(
        f"#!/bin/sh\necho '{bkill_stdout}'\n"
        f"echo '{bkill_stderr}' >&2\n"
        f"echo $@ > 'bkill_args'\n"
        f"exit {bkill_returncode}",
        encoding="utf-8",
    )
    bkill_path.chmod(bkill_path.stat().st_mode | stat.S_IEXEC)

    driver = LsfDriver()

    driver._iens2jobid = mocked_iens2jobid
    await driver.kill(iens_to_kill)
    if expected_logged_error:
        assert expected_logged_error in caplog.text
    else:
        assert (
            mocked_iens2jobid[iens_to_kill]
            == Path("bkill_args").read_text(encoding="utf-8").strip()
        )
