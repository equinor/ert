import asyncio
import logging
import os
import random
import stat
import string
import sys
from contextlib import ExitStack as does_not_raise
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ert.scheduler import SlurmDriver
from ert.scheduler.slurm_driver import _seconds_to_slurm_time_format
from tests.ert.utils import poll

from .conftest import mock_bin


def nonempty_string_without_whitespace():
    return st.text(
        st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P")), min_size=1
    )


@pytest.fixture
def capturing_sbatch(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    bin_path = Path("bin")
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    sbatch_path = bin_path / "sbatch"
    sbatch_path.write_text(
        "#!/bin/sh\n" "echo $@ > captured_sbatch_args\n" "echo 1",
        encoding="utf-8",
    )
    sbatch_path.chmod(sbatch_path.stat().st_mode | stat.S_IEXEC)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "sbatch_script, scontrol_script, exit_code",
    [
        pytest.param("echo 0", 'echo "JobState=COMPLETED ExitCode=0:0"', 0),
    ],
)
async def test_exit_codes(tmp_path_factory, sbatch_script, scontrol_script, exit_code):
    tmp_path = tmp_path_factory.mktemp("exit_codes")

    mocked_scontrol = tmp_path / "scontrol"
    mocked_scontrol.write_text(f"#!/bin/sh\n{scontrol_script}")
    mocked_scontrol.chmod(mocked_scontrol.stat().st_mode | stat.S_IEXEC)

    mocked_sbatch = tmp_path / "sbatch"
    mocked_sbatch.write_text(f"#!/bin/sh\n{sbatch_script}")
    mocked_sbatch.chmod(mocked_sbatch.stat().st_mode | stat.S_IEXEC)

    driver = SlurmDriver(sbatch_cmd=mocked_sbatch, scontrol_cmd=mocked_scontrol)
    await driver.submit(0, 'echo "hello"')

    assert await driver._get_exit_code("0") == exit_code


@pytest.mark.usefixtures("capturing_sbatch")
async def test_submit_sets_out():
    driver = SlurmDriver()
    await driver.submit(0, "sleep", name="myjobname")
    assert "--output=myjobname.stdout" in Path("captured_sbatch_args").read_text(
        encoding="utf-8"
    )
    assert "--error=myjobname.stderr" in Path("captured_sbatch_args").read_text(
        encoding="utf-8"
    )


@pytest.mark.usefixtures("capturing_sbatch")
@given(num_cpu=st.integers(min_value=1))
async def test_numcpu_sets_ntasks(num_cpu):
    driver = SlurmDriver()
    await driver.submit(0, "sleep", name="myjobname", num_cpu=num_cpu)
    assert f"--ntasks={num_cpu}" in Path("captured_sbatch_args").read_text(
        encoding="utf-8"
    )


@pytest.mark.usefixtures("capturing_sbatch")
@given(memory_in_mb=st.integers(min_value=1))
async def test_memory_is_set(memory_in_mb):
    driver = SlurmDriver(memory=f"{memory_in_mb}mb")
    await driver.submit(0, "sleep", name="myjobname")
    assert f"--mem={memory_in_mb}mb" in Path("captured_sbatch_args").read_text(
        encoding="utf-8"
    )


@pytest.mark.usefixtures("capturing_sbatch")
@given(memory_in_bytes=st.integers(min_value=1))
async def test_realization_memoryt(memory_in_bytes):
    driver = SlurmDriver(realization_memory=memory_in_bytes)
    await driver.submit(0, "sleep", name="myjobname")
    assert f"--mem={memory_in_bytes // 1024**2}M" in Path(
        "captured_sbatch_args"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "memory_value, realization_memory_value, expectation",
    [
        # Emptry string for memory is equivalent to zero integer for realiation_memory
        ("", 1, does_not_raise()),
        ("1mb", 0, does_not_raise()),
        ("1mb", 1, pytest.raises(ValueError)),
        ("0", 1, pytest.raises(ValueError)),
    ],
)
async def test_overspecified_memory_allocation(
    memory_value, realization_memory_value, expectation
):
    # Emptry string for memory is equivalent to zero integer for realiation_memory
    with expectation:
        SlurmDriver(memory=memory_value, realization_memory=realization_memory_value)


@pytest.mark.usefixtures("capturing_sbatch")
@given(memory_per_cpu_in_mb=st.integers(min_value=1))
async def test_memory_per_cpu_is_set(memory_per_cpu_in_mb):
    driver = SlurmDriver(memory_per_cpu=f"{memory_per_cpu_in_mb}mb")
    await driver.submit(0, "sleep", name="myjobname")
    assert f"--mem-per-cpu={memory_per_cpu_in_mb}mb" in Path(
        "captured_sbatch_args"
    ).read_text(encoding="utf-8")


@pytest.mark.usefixtures("capturing_sbatch")
@given(exclude=st.text(st.characters(whitelist_categories=("Lu",)), min_size=1))
async def test_exclude_is_set(exclude):
    driver = SlurmDriver(exclude_hosts=f"{exclude}")
    await driver.submit(0, "sleep", name="myjobname")
    assert f"--exclude={exclude}" in Path("captured_sbatch_args").read_text(
        encoding="utf-8"
    )


@pytest.mark.usefixtures("capturing_sbatch")
@given(include=st.text(st.characters(whitelist_categories=("Lu",)), min_size=1))
async def test_include_is_set(include):
    driver = SlurmDriver(include_hosts=f"{include}")
    await driver.submit(0, "sleep", name="myjobname")
    assert f"--nodelist={include}" in Path("captured_sbatch_args").read_text(
        encoding="utf-8"
    )


@pytest.mark.usefixtures("capturing_sbatch")
@given(queue_name=st.text(st.characters(whitelist_categories=("Lu",)), min_size=1))
async def test_queue_name_is_set(queue_name):
    driver = SlurmDriver(queue_name=f"{queue_name}")
    await driver.submit(0, "sleep", name="myjobname")
    assert f"--partition={queue_name}" in Path("captured_sbatch_args").read_text(
        encoding="utf-8"
    )


@pytest.mark.usefixtures("capturing_sbatch")
@given(project_code=st.text(st.characters(whitelist_categories=("Lu",)), min_size=1))
async def test_project_code_is_set(project_code):
    driver = SlurmDriver(project_code=f"{project_code}")
    await driver.submit(0, "sleep", name="myjobname")
    assert f"--account={project_code}" in Path("captured_sbatch_args").read_text(
        encoding="utf-8"
    )


@pytest.mark.usefixtures("capturing_sbatch")
@given(max_runtime=st.floats(min_value=1, max_value=999999999))
async def test_max_runtime_is_properly_formatted(max_runtime):
    # According to https://slurm.schedmd.com/sbatch.html we accept the formats
    # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours",
    # "days-hours:minutes" and "days-hours:minutes:seconds".
    driver = SlurmDriver(max_runtime=max_runtime)
    await driver.submit(0, "sleep", name="myjobname")
    cmd_args = Path("captured_sbatch_args").read_text(encoding="utf-8").split()
    time_argument = next(
        arg.split("=")[1] for arg in cmd_args if arg.startswith("--time")
    )
    if "-" in time_argument:
        int(time_argument.split("-")[0])
        hhmmss = time_argument.split("-")[1]
    else:
        hhmmss = time_argument
    hh, mm, ss = map(int, hhmmss.split(":"))
    assert 0 <= hh < 24
    assert 0 <= mm < 60
    assert 0 <= ss < 60


@pytest.mark.usefixtures("capturing_sbatch")
@pytest.mark.parametrize("float_seconds", [0.0, 0.1, 0.99])
async def test_driver_will_ignore_max_runtime_less_than_1_seconds(float_seconds):
    driver = SlurmDriver(max_runtime=float_seconds)
    await driver.submit(0, "sleep", name="skip_low_max_runtime")
    assert "--time" not in Path("captured_sbatch_args").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "seconds, expected_format",
    [
        (1, "0:00:01"),
        (1.9, "0:00:01"),
        (61, "0:01:01"),
        (60 * 60, "1:00:00"),
        (24 * 60 * 60 - 1, "23:59:59"),
        (24 * 60 * 60, "1-0:00:00"),
    ],
)
async def test_max_runtime_formatting_samples(seconds, expected_format):
    assert _seconds_to_slurm_time_format(seconds) == expected_format


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "sbatch_script, expectation",
    [
        pytest.param(
            "echo '444'",
            does_not_raise(),
            id="all-good",
        ),
        pytest.param(
            "echo '0'",
            does_not_raise(),
            id="zero_job_id",
        ),
        pytest.param("exit 1", pytest.raises(RuntimeError), id="plain_exit_1"),
        pytest.param(
            "exit 0", pytest.raises(RuntimeError), id="exit_0_but_empty_stdout"
        ),
    ],
)
async def test_faulty_sbatch(monkeypatch, tmp_path, sbatch_script, expectation):
    monkeypatch.chdir(tmp_path)
    bin_path = Path("bin")
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    sbatch_path = bin_path / "sbatch"
    sbatch_path.write_text(f"#!/bin/sh\n{sbatch_script}")
    sbatch_path.chmod(sbatch_path.stat().st_mode | stat.S_IEXEC)
    driver = SlurmDriver()
    with expectation:
        await driver.submit(0, "sleep")


async def test_faulty_sbatch_produces_error_log(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    bin_path = Path("bin")
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")

    _out = "THIS_IS_OUTPUT"
    _err = "THIS_IS_ERROR"
    sbatch_script = f"echo {_out} && echo {_err} >&2; exit 1"
    sbatch_path = bin_path / "sbatch"
    sbatch_path.write_text(f"#!/bin/sh\n{sbatch_script}")
    sbatch_path.chmod(sbatch_path.stat().st_mode | stat.S_IEXEC)

    driver = SlurmDriver()
    with pytest.raises(RuntimeError):
        await driver.submit(0, "sleep")
    assert (
        f'failed with exit code 1, output: "{_out}", and error: "{_err}"'
        in driver._job_error_message_by_iens[0]
    )


async def test_kill_before_submit_logs_error(caplog):
    driver = SlurmDriver()
    await driver.kill(0)
    assert "ERROR" in caplog.text
    assert "realization 0 has never been submitted" in caplog.text


@pytest.fixture(autouse=True)
def mock_slurm(pytestconfig, monkeypatch, tmp_path):
    if pytestconfig.getoption("slurm"):
        # User provided --slurm, which means we should use an actual Slurm
        # cluster without mocking anything.""
        return
    mock_bin(monkeypatch, tmp_path)


async def test_slurm_stdout_file(tmp_path, job_name):
    os.chdir(tmp_path)
    driver = SlurmDriver()
    await driver.submit(0, "sh", "-c", "echo yay", name=job_name)
    await poll(driver, {0})
    slurm_stdout = Path(f"{job_name}.stdout").read_text(encoding="utf-8")
    assert Path(f"{job_name}.stdout").exists(), "Slurm system did not write output file"
    assert "yay" in slurm_stdout


async def test_slurm_dumps_stderr_to_file(tmp_path, job_name):
    os.chdir(tmp_path)
    driver = SlurmDriver()
    failure_message = "failURE"
    await driver.submit(0, "sh", "-c", f"echo {failure_message} >&2", name=job_name)
    await poll(driver, {0})
    assert Path(f"{job_name}.stderr").exists(), "Slurm system did not write stderr file"

    assert (
        Path(f"{job_name}.stderr").read_text(encoding="utf-8").strip()
        == failure_message
    )


def generate_random_text(size):
    letters = string.ascii_letters
    return "".join(random.choice(letters) for _ in range(size))


@pytest.mark.parametrize("tail_chars_to_read", [(5), (50), (500), (700)])
async def test_slurm_can_retrieve_stdout_and_stderr(
    tmp_path, job_name, tail_chars_to_read
):
    os.chdir(tmp_path)
    driver = SlurmDriver()
    num_written_characters = 600
    _out = generate_random_text(num_written_characters)
    _err = generate_random_text(num_written_characters)
    await driver.submit(0, "sh", "-c", f"echo {_out} && echo {_err} >&2", name=job_name)
    await poll(driver, {0})
    message = driver.read_stdout_and_stderr_files(
        runpath=".",
        job_name=job_name,
        num_characters_to_read_from_end=tail_chars_to_read,
    )

    stderr_txt = Path(f"{job_name}.stderr").read_text(encoding="utf-8").strip()
    stdout_txt = Path(f"{job_name}.stdout").read_text(encoding="utf-8").strip()

    assert stderr_txt[-min(tail_chars_to_read, num_written_characters) + 2 :] in message
    assert stdout_txt[-min(tail_chars_to_read, num_written_characters) + 2 :] in message


@pytest.mark.integration_test
async def test_submit_to_named_queue(tmp_path, job_name):
    """If the environment variable _ERT_TEST_ALTERNATIVE_QUEUE is defined
    a job will be attempted submitted to that queue.

    * Note that what is called a "queue" in Ert is a "partition" in Slurm lingo.

    As Ert does not keep track of which queue a job is executed in, we can only
    test for success for the job."""
    os.chdir(tmp_path)
    driver = SlurmDriver(queue_name=os.getenv("_ERT_TESTS_ALTERNATIVE_QUEUE"))
    await driver.submit(0, "sh", "-c", f"echo test > {tmp_path}/test", name=job_name)
    await poll(driver, {0})

    assert (tmp_path / "test").read_text(encoding="utf-8") == "test\n"


@pytest.mark.usefixtures("use_tmpdir")
async def test_submit_with_num_cpu(pytestconfig, job_name):
    if not pytestconfig.getoption("slurm"):
        return

    num_cpu = 2
    driver = SlurmDriver()
    await driver.submit(0, "sh", "-c", "echo test>test", name=job_name, num_cpu=num_cpu)
    job_id = driver._iens2jobid[0]
    await poll(driver, {0})

    process = await asyncio.create_subprocess_exec(
        "scontrol",
        "show",
        "job",
        job_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    assert " NumCPUs=2 " in stdout.decode(
        errors="ignore"
    ), f"Could not verify processor allocation from stdout: {stdout}, stderr: {stderr}"

    assert Path("test").read_text(encoding="utf-8") == "test\n"


@pytest.mark.integration_test
@pytest.mark.flaky(reruns=3)
async def test_kill_before_submit_is_finished(
    tmp_path, monkeypatch, caplog, pytestconfig
):
    os.chdir(tmp_path)

    if pytestconfig.getoption("slurm"):
        # Allow more time when tested on a real compute cluster to avoid false positives.
        job_kill_window = 5
        test_grace_time = 10
    elif sys.platform.startswith("darwin"):
        # Mitigate flakiness on low-power test nodes
        job_kill_window = 5
        test_grace_time = 10
    else:
        job_kill_window = 1
        test_grace_time = 2

    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    monkeypatch.setenv("PATH", f"{bin_path}:{os.environ['PATH']}")
    sbatch_path = bin_path / "slow_sbatch"
    sbatch_path.write_text(
        "#!/bin/sh\nsleep 0.1\nsbatch $@",
        encoding="utf-8",
    )
    sbatch_path.chmod(sbatch_path.stat().st_mode | stat.S_IEXEC)

    caplog.set_level(logging.DEBUG)
    driver = SlurmDriver(sbatch_cmd="slow_sbatch")

    # Allow submit and kill to be interleaved by asyncio by issuing
    # submit() in its own asyncio Task:
    asyncio.create_task(
        driver.submit(
            # The sleep is the time window in which we can kill the job before
            # the unwanted finish message appears on disk.
            0,
            "sh",
            "-c",
            f"sleep {job_kill_window}; touch {tmp_path}/survived",
        )
    )
    await asyncio.sleep(0.01)  # Allow submit task to start executing
    await driver.kill(0)  # This will wait until the submit is done and then kill

    async def finished(iens: int, returncode: int):
        assert iens == 0
        # Slurm assigns returncode 0 even when they are killed.
        assert returncode == 0

    await poll(driver, {0}, finished=finished)

    # In case the return value of the killed job is correct but the submitted
    # shell script is still running for whatever reason, a file called
    # "survived" will appear on disk. Wait for it, and then ensure it is not
    # there.
    assert test_grace_time > job_kill_window, "Wrong test setup"
    await asyncio.sleep(test_grace_time)
    assert not Path(
        "survived"
    ).exists(), "The process children of the job should also have been killed"
