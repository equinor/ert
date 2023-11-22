import logging
import os
import re
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest

from ert.__main__ import ert_parser
from ert.cli import ENSEMBLE_EXPERIMENT_MODE
from ert.cli.main import run_cli
from ert.job_queue.driver import LSFDriver


@pytest.fixture
def mock_bsub(tmp_path):
    script_path = tmp_path / "mock_bsub"
    script_path.write_text(
        "#!/usr/bin/env python3"
        + dedent(
            """
                import sys
                import time
                import random
                run_path = sys.argv[-1]
                with open("job_paths", "a+", encoding="utf-8") as jobs_file:
                    jobs_file.write(f"{run_path}\\n")

                # debug purposes
                with open("bsub_log", "a+", encoding="utf-8") as f:
                    f.write(f"{str(sys.argv)}\\n")

                time.sleep(0.5)


                if "exit.sh" in sys.argv:
                    exit(1)


                if "invalid_lsf_id.sh" in sys.argv:
                    print(1)
                else:
                    _id = str(random.randint(0, 10000000))
                    print(f"Job <{_id}> is submitted to default queue <normal>.")
            """
        )
    )
    os.chmod(script_path, 0o755)


@pytest.fixture
def mock_bkill(tmp_path):
    script_path = tmp_path / "mock_bkill"
    script_path.write_text(
        "#!/usr/bin/env python3"
        + dedent(
            """
                import sys
                import time
                import random
                job_id = sys.argv[-1]
                with open("job_ids", "a+", encoding="utf-8") as jobs_file:
                    jobs_file.write(f"{job_id}\\n")


                time.sleep(0.5)


                if job_id == "non_existent_jobid":
                    print(f"bkill: jobid {job_id} not found")
                    exit(1)
            """
        )
    )
    os.chmod(script_path, 0o755)


@pytest.fixture
def mock_bjobs(tmp_path):
    script = "#!/usr/bin/env python3" + dedent(
        """
        import datetime
        import json
        import os.path
        import sys

           # Just to have a log for test purposes what is actually thrown
           # towards the bjobs command
        with open("bjobs_log", "a+", encoding="utf-8") as f:
            f.write(f"{str(sys.argv)}\\n")
        print("JOBID\tUSER\tSTAT\tQUEUE\tFROM_HOST\tEXEC_HOST\tJOB_NAME\tSUBMIT_TIME")

            # Statuses LSF can give us
            # "PEND"
            # "SSUSP"
            # "PSUSP"
            # "USUSP"
            # "RUN"
            # "EXIT"
            # "ZOMBI" : does not seem to be available from the api.
            # "DONE"
            # "PDONE" : Post-processor is done.
            # "UNKWN"

        with open("mocked_result", mode="r+", encoding="utf-8") as result_line:
            result = result_line.read()
        if "exit" in result.split("\t"):
            exit(1)
        print(result)


           """
    )
    script_path = tmp_path / "mock_bjobs"
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(script)

    os.chmod(script_path, 0o755)


class MockRealizationState:
    def verify_kill(self):
        print("Realization was verified killed")


def create_fake_bjobs_result(dir: str, job_id: str, status: str):
    # print("JOBID USER STAT QUEUE FROM_HOST EXEC_HOST JOB_NAME SUBMIT_TIME")
    Path(dir / "mocked_result").write_text(
        f"{job_id}\tpytest\t{status}\ttest_queue\thost1\thost2\ttest_job\t{str(datetime.now())}"
    )


async def test_submit_failure_script_exit(mock_bsub, caplog, tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    lsf_driver = LSFDriver(None)
    lsf_driver.set_option("BSUB_CMD", tmpdir / "mock_bsub")
    lsf_driver._max_attempt = 3
    lsf_driver._SLEEP_PERIOD = 0.2
    with patch(
        "ert.job_queue.realization_state.QueueableRealization"
    ) as mock_realization:
        mock_realization.realization.run_arg.iens.return_value = 0
        mock_realization.realization.job_script = "exit.sh"
        mock_realization.realization.run_arg.runpath = "/usr/random/ert_path"
        with pytest.raises(
            RuntimeError, match="Maximum number of submit errors exceeded"
        ):
            await lsf_driver.submit(mock_realization)

    job_paths = Path("job_paths").read_text(encoding="utf-8").strip().split("\n")

    # should try command 3 times before exiting
    assert len(job_paths) == 3

    output = caplog.text
    assert re.search("Tried submitting job 3 times, but it still failed", output)
    assert len(re.findall("returned non-zero exitcode: 1", output)) == 3


async def test_submit_success(mock_bsub, caplog, tmpdir, monkeypatch):
    caplog.set_level(logging.DEBUG)
    monkeypatch.chdir(tmpdir)
    lsf_driver = LSFDriver(None)
    lsf_driver.set_option("BSUB_CMD", tmpdir / "mock_bsub")
    lsf_driver._max_attempt = 3

    with patch(
        "ert.job_queue.realization_state.QueueableRealization"
    ) as mock_realization:
        mock_realization.realization.run_arg.iens.return_value = 0
        mock_realization.realization.job_script = "valid_script.sh"
        mock_realization.realization.run_arg.runpath = "/usr/random/ert_path"
        await lsf_driver.submit(mock_realization)

    job_paths = Path("job_paths").read_text(encoding="utf-8").strip().split("\n")
    assert len(job_paths) == 1

    output = caplog.text
    print(f"{output=}")
    assert re.search("Submitted job.*and got LSF JOBID", output)
    assert re.search("submitted to default queue", output)


async def test_poll_statuses_while_already_polling(
    mock_bjobs, caplog, tmpdir, monkeypatch
):
    monkeypatch.chdir(tmpdir)

    lsf_driver = LSFDriver(None)
    lsf_driver._currently_polling = True
    lsf_driver._statuses.update({"test_lsf_job_id": "RUNNING"})
    statuses: dict = await lsf_driver.poll_statuses()
    assert statuses["test_lsf_job_id"] == "RUNNING"
    # Should never call bjobs
    assert not Path("bjobs_logs").exists()

    output = caplog.text
    assert output == ""
    assert lsf_driver._currently_polling


async def test_poll_statuses_before_submitting_jobs(
    mock_bjobs, caplog, tmpdir, monkeypatch
):
    monkeypatch.chdir(tmpdir)

    lsf_driver = LSFDriver(None)
    lsf_driver.set_option("BJOBS_CMD", tmpdir / "mock_bjobs")

    # should print out and ignore the unknown job id
    await lsf_driver.poll_statuses()

    assert not Path("bjobs_logs").exists()

    output = caplog.text
    assert re.search("Skipped polling due to no jobs submitted", output)


async def test_poll_statuses_bjobs_exit_code_1(mock_bjobs, caplog, tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)

    lsf_driver = LSFDriver(None)
    lsf_driver.set_option("BJOBS_CMD", tmpdir / "mock_bjobs")
    lsf_driver._max_attempt = 3

    # will return a job id triggering exit(1) in bjobs
    create_fake_bjobs_result(tmpdir, job_id="exit", status="PEND")

    mock_realization_state = MockRealizationState()
    lsf_driver._realstate_to_lsfid[mock_realization_state] = "valid_job_id"
    lsf_driver._lsfid_to_realstate["valid_job_id"] = mock_realization_state

    # should print out and ignore the unknown job id
    await lsf_driver.poll_statuses()

    bjobs_logs = Path("bjobs_log").read_text(encoding="utf-8").strip().split("\n")

    # Should only call bjobs once
    assert len(bjobs_logs) == 3

    output = caplog.text
    assert len(re.findall("bjobs returned non-zero exitcode: 1", output)) == 3


async def test_poll_statuses_bjobs_returning_unknown_job_id(
    mock_bjobs, caplog, tmpdir, monkeypatch
):
    monkeypatch.chdir(tmpdir)

    lsf_driver = LSFDriver(None)
    lsf_driver.set_option("BJOBS_CMD", tmpdir / "mock_bjobs")
    lsf_driver._max_attempt = 3
    lsf_driver._SLEEP_PERIOD = 0.2
    # will return a job id not belonging to this run
    create_fake_bjobs_result(tmpdir, job_id="unknown_job_id", status="PEND")

    mock_realization_state = MockRealizationState()
    lsf_driver._realstate_to_lsfid[mock_realization_state] = "valid_job_id"
    lsf_driver._lsfid_to_realstate["valid_job_id"] = mock_realization_state

    # should print out and ignore the unknown job id
    await lsf_driver.poll_statuses()

    bjobs_logs = Path("bjobs_log").read_text(encoding="utf-8").strip().split("\n")

    # Should only call bjobs once
    assert len(bjobs_logs) == 1

    output = caplog.text
    print(f"{output=}")
    assert re.search(r"Found unknown job id \(unknown_job_id\)", output)


async def test_poll_statuses_bjobs_returning_unrecognized_status(
    mock_bjobs, tmpdir, monkeypatch
):
    monkeypatch.chdir(tmpdir)

    lsf_driver = LSFDriver(None)
    lsf_driver.set_option("BJOBS_CMD", tmpdir / "mock_bjobs")
    lsf_driver._max_attempt = 3
    lsf_driver._SLEEP_PERIOD = 0.2
    create_fake_bjobs_result(tmpdir, job_id="valid_job_id", status="EATING")

    mock_realization_state = MockRealizationState()
    lsf_driver._realstate_to_lsfid[mock_realization_state] = "valid_job_id"
    lsf_driver._lsfid_to_realstate["valid_job_id"] = mock_realization_state

    with pytest.raises(
        RuntimeError,
        match="The lsf_status EATING for job valid_job_id was not recognized",
    ):
        await lsf_driver.poll_statuses()

    bjobs_logs = Path("bjobs_log").read_text(encoding="utf-8").strip().split("\n")

    # Should only call bjobs once
    assert len(bjobs_logs) == 1


async def test_kill_bkill_non_existent_jobid_exit_code_1(
    mock_bkill, caplog, tmpdir, monkeypatch
):
    monkeypatch.chdir(tmpdir)
    lsf_driver = LSFDriver(None)
    lsf_driver.set_option("BKILL_CMD", tmpdir / "mock_bkill")
    lsf_driver._max_attempt = 3
    lsf_driver._SLEEP_PERIOD = 0.2
    mock_realization_state = MockRealizationState()
    lsf_driver._realstate_to_lsfid[mock_realization_state] = "non_existent_jobid"

    with pytest.raises(RuntimeError, match="Maximum number of kill errors exceeded"):
        await lsf_driver.kill(mock_realization_state)

    output = caplog.text
    out_log = output.split("\n")
    job_ids_from_file = Path("job_ids").read_text(encoding="utf-8").strip().split("\n")
    assert len(job_ids_from_file) == lsf_driver._max_attempt
    print(f"{out_log=}")
    assert (
        len(re.findall("bkill: jobid non_existent_jobid not found", output))
        == lsf_driver._max_attempt
    )
    assert re.search(
        "Tried killing job non_existent_jobid 3 times, but it still failed", output
    )
    assert (
        len(re.findall("returned non-zero exitcode: 1", output))
        == lsf_driver._max_attempt
    )


@pytest.fixture
def copy_lsf_poly_case(copy_poly_case, tmp_path):
    # Overwriting the "poly.ert" config file in tmpdir runpath
    # with our own customized config with at least sets queue option to LSF and
    # introducing the mocked jobs.

    config = [
        "JOBNAME poly_%d\n",
        "QUEUE_SYSTEM  LSF\n",
        "QUEUE_OPTION LSF MAX_RUNNING 10\n",
        f"QUEUE_OPTION LSF BJOBS_CMD {tmp_path}/mock_bjobs\n",
        f"QUEUE_OPTION LSF BSUB_CMD {tmp_path}/mock_bsub\n",
        "RUNPATH poly_out/realization-<IENS>/iter-<ITER>\n",
        "OBS_CONFIG observations\n",
        "NUM_REALIZATIONS 10\n",
        "MIN_REALIZATIONS 1\n",
        "GEN_KW COEFFS coeff_priors\n",
        "GEN_DATA POLY_RES RESULT_FILE:poly.out\n",
        "INSTALL_JOB poly_eval POLY_EVAL\n",
        "SIMULATION_JOB poly_eval\n",
    ]
    with open("poly.ert", "w", encoding="utf-8") as fh:
        fh.writelines(config)


@pytest.mark.skip(reason="Integration Test - does not work with the new python driver")
@pytest.mark.usefixtures(
    "copy_lsf_poly_case",
    "mock_bsub",
    "mock_bjobs",
    "mock_start_server",
)
@pytest.mark.integration_test
def test_run_mocked_lsf_queue():
    run_cli(
        ert_parser(
            ArgumentParser(prog="test_main"),
            [
                ENSEMBLE_EXPERIMENT_MODE,
                "poly.ert",
            ],
        )
    )
    log = Path("bsub_log").read_text(encoding="utf-8")
    for i in range(10):
        assert f"'-J', 'poly_{i}'" in log
