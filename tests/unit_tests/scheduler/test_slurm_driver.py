import os
import stat
from contextlib import ExitStack as does_not_raise
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ert.scheduler import SlurmDriver


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
@given(max_runtime=st.integers(min_value=1))
async def test_max_runtime_is_set(max_runtime):
    driver = SlurmDriver(max_runtime=str(max_runtime))
    await driver.submit(0, "sleep", name="myjobname")
    assert f"--time={max_runtime}" in Path("captured_sbatch_args").read_text(
        encoding="utf-8"
    )


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
