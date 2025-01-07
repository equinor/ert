import asyncio
import os
import signal
from pathlib import Path
from textwrap import dedent


def create_ert_config(path: Path):
    ert_config_path = Path(path / "ert_config.ert")
    Path(path / "TEST_JOB").write_text("EXECUTABLE test_script.sh", encoding="utf-8")
    Path(path / "test_script.sh").write_text(
        dedent(
            """\
            #!/bin/sh
            echo $$ > forward_model_pid
            sleep 20
            """
        ),
        encoding="utf-8",
    )
    os.chmod(path / "test_script.sh", 0o755)
    ert_config_path.write_text(
        dedent(
            r"""
            JOBNAME test_%d
            QUEUE_SYSTEM LOCAL
            QUEUE_OPTION LOCAL MAX_RUNNING 50
            RUNPATH test_out/realization-<IENS>/iter-<ITER>
            NUM_REALIZATIONS 1
            MIN_REALIZATIONS 1
            INSTALL_JOB write_pid_to_file_and_sleep TEST_JOB
            FORWARD_MODEL write_pid_to_file_and_sleep
            """
        ),
        encoding="utf-8",
    )


async def test_subprocesses_live_on_after_ert_dies(tmp_path):
    # Have ERT run a forward model that writes in PID to a file, then sleeps
    # Forcefully terminate ERT and assert that the child process is not terminated
    create_ert_config(tmp_path)

    ert_process = await asyncio.create_subprocess_exec(
        "ert", "test_run", "ert_config.ert", cwd=tmp_path
    )
    check_path_retry, check_path_max_retries = 0, 200
    expected_path = Path(tmp_path, "test_out/realization-0/iter-0/forward_model_pid")
    while not expected_path.exists() and check_path_retry < check_path_max_retries:
        check_path_retry += 1
        await asyncio.sleep(0.5)

    assert ert_process.returncode is None
    assert expected_path.exists()
    child_process_id = expected_path.read_text(encoding="utf-8").strip()
    assert child_process_id

    assert ert_process.returncode is None
    ert_process.kill()
    await ert_process.wait()
    assert ert_process.returncode is not None

    # Child process should still exist
    ps_process = await asyncio.create_subprocess_exec("ps", "-p", child_process_id)
    assert await ps_process.wait() == 0

    # Clean up the child process
    os.killpg(os.getpgid(int(child_process_id)), signal.SIGKILL)
