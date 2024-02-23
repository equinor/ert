import os
import stat
from pathlib import Path

import pytest

from ert.scheduler import OpenPBSDriver


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
