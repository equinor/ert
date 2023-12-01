import logging
import os
import re
import stat
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from threading import BoundedSemaphore
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from ert.__main__ import ert_parser
from ert.cli import ENSEMBLE_EXPERIMENT_MODE
from ert.cli.main import run_cli
from ert.config import QueueSystem
from ert.run_arg import RunArg
from ert.scheduler import Driver, QueueableRealization
from ert.scheduler.driver import LSFDriver
from ert.storage import EnsembleAccessor

DUMMY_CONFIG: Dict[str, Any] = {
    "job_script": "job_script.py",
    "num_cpu": 1,
    "job_name": "dummy_job_{}",
    "run_path": "dummy_path_{}",
}
MOCK_BSUB = """#!/bin/sh
echo "$@" > test.out
"""
"""A dummy bsub script that instead of submitting a job to an LSF cluster
writes the arguments it got to a file called test.out, mimicking what
an actual cluster node might have done."""


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
                    f.write(f"{' '.join(sys.argv)}\\n")

                time.sleep(0.5)

                if "exit.sh" in sys.argv:
                    exit(1)

                if "gargled_return.sh" in sys.argv:
                    print("wait,this_is_not_a_valid_return_format")
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


class MockStateHandler:
    id = "SUBMITTED"


class MockRunArg:
    runpath = "/usr/random/ert_path"

    def iens():
        return 0


class MockQueueableRealization:
    run_arg = MockRunArg()


class MockRealizationState:
    _state = "SUBMITTED"
    _verified_killed = False
    realization = MockQueueableRealization()
    current_state = MockStateHandler()

    def verify_kill(self):
        self._verified_killed = True
        print("Realization was verified killed")

    def accept(self):
        self.current_state.id = "PENDING"

    def start(self):
        self.current_state.id = "RUNNING"

    def runend(self):
        self.current_state.id = "DONE"


def create_fake_bjobs_result(dir: str, job_id: str, status: str):
    # print("JOBID USER STAT QUEUE FROM_HOST EXEC_HOST JOB_NAME SUBMIT_TIME")
    Path(dir / "mocked_result").write_text(
        f"{job_id}\tpytest\t{status}\ttest_queue\thost1\thost2\ttest_job\t{str(datetime.now())}"
    )


@pytest.mark.asyncio
async def test_submit_failure_script_exit(mock_bsub, caplog, tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    lsf_driver = LSFDriver(None)
    lsf_driver = LSFDriver(queue_options=[("BSUB_CMD", tmpdir / "mock_bsub")])
    lsf_driver._max_attempt = 3
    lsf_driver._retry_sleep_period = 0

    mock_realization_state = MockRealizationState()
    mock_realization_state.realization.job_script = "exit.sh"

    with pytest.raises(RuntimeError, match="Maximum number of submit errors exceeded"):
        await lsf_driver.submit(mock_realization_state)

    job_paths = Path("job_paths").read_text(encoding="utf-8").strip().split("\n")

    # should try command 3 times before exiting
    assert len(job_paths) == 3

    output = caplog.text
    assert len(re.findall("bsub returned non-zero exitcode: 1", output)) == 3


@pytest.mark.asyncio
async def test_submit_failure_badly_formated_return(
    mock_bsub, caplog, tmpdir, monkeypatch
):
    monkeypatch.chdir(tmpdir)
    lsf_driver = LSFDriver(queue_options=[("BSUB_CMD", tmpdir / "mock_bsub")])
    lsf_driver._max_attempt = 3
    lsf_driver._retry_sleep_period = 0

    mock_realization_state = MockRealizationState()
    mock_realization_state.realization.job_script = "gargled_return.sh"

    with pytest.raises(RuntimeError, match="Maximum number of submit errors exceeded"):
        await lsf_driver.submit(mock_realization_state)

    job_paths = Path("job_paths").read_text(encoding="utf-8").strip().split("\n")

    # should try command 3 times before exiting
    assert len(job_paths) == 3

    output = caplog.text
    print(f"{output=}")
    assert len(re.findall("Could not parse lsf id from", output)) == 3


@pytest.mark.asyncio
async def test_submit_success(mock_bsub, caplog, tmpdir, monkeypatch):
    caplog.set_level(logging.DEBUG)
    monkeypatch.chdir(tmpdir)
    lsf_driver = LSFDriver(queue_options=[("BSUB_CMD", tmpdir / "mock_bsub")])
    lsf_driver._max_attempt = 3

    mock_realization_state = MockRealizationState()
    mock_realization_state.realization.job_script = "valid_script.sh"

    await lsf_driver.submit(mock_realization_state)

    job_paths = Path("job_paths").read_text(encoding="utf-8").strip().split("\n")
    assert len(job_paths) == 1
    output = caplog.text
    assert re.search("Submitted job.*and got LSF JOBID", output)
    assert re.search("submitted to default queue", output)
    assert mock_realization_state.current_state.id == "PENDING"


@pytest.mark.asyncio
async def test_poll_statuses_while_already_polling(
    mock_bjobs, caplog, tmpdir, monkeypatch
):
    monkeypatch.chdir(tmpdir)

    lsf_driver = LSFDriver(None)
    lsf_driver._currently_polling = True

    await lsf_driver.poll_statuses()

    # Should not call bjobs
    assert not Path("bjobs_logs").exists()

    output = caplog.text
    assert output == ""
    assert lsf_driver._currently_polling


@pytest.mark.asyncio
async def test_poll_statuses_before_submitting_jobs():
    lsf_driver = LSFDriver(None)

    # should not crash
    await lsf_driver.poll_statuses()


@pytest.mark.asyncio
async def test_poll_statuses_bjobs_exit_code_1(mock_bjobs, caplog, tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)

    lsf_driver = LSFDriver(queue_options=[("BJOBS_CMD", tmpdir / "mock_bjobs")])
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
    assert (
        mock_realization_state.current_state.id
        == MockRealizationState().current_state.id
    )


@pytest.mark.asyncio
async def test_poll_statuses_bjobs_returning_unknown_job_id(
    mock_bjobs, tmpdir, monkeypatch
):
    monkeypatch.chdir(tmpdir)

    lsf_driver = LSFDriver(queue_options=[("BJOBS_CMD", tmpdir / "mock_bjobs")])
    lsf_driver._max_attempt = 3
    lsf_driver._retry_sleep_period = 0
    # will return a job id not belonging to this run
    create_fake_bjobs_result(tmpdir, job_id="unknown_job_id", status="PEND")

    mock_realization_state = MockRealizationState()
    lsf_driver._realstate_to_lsfid[mock_realization_state] = "valid_job_id"
    lsf_driver._lsfid_to_realstate["valid_job_id"] = mock_realization_state

    # Should ignore the unknown job id
    await lsf_driver.poll_statuses()

    bjobs_logs = Path("bjobs_log").read_text(encoding="utf-8").strip().split("\n")

    # Should only call bjobs once
    assert len(bjobs_logs) == 1
    assert (
        mock_realization_state.current_state.id == MockRealizationState.current_state.id
    )


@pytest.mark.asyncio
async def test_poll_statuses_bjobs_returning_unrecognized_status(
    mock_bjobs, tmpdir, monkeypatch
):
    monkeypatch.chdir(tmpdir)

    lsf_driver = LSFDriver(queue_options=[("BJOBS_CMD", tmpdir / "mock_bjobs")])
    lsf_driver._max_attempt = 3
    lsf_driver._retry_sleep_period = 0
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


@pytest.mark.asyncio
async def test_poll_statuses_bjobs_returning_updated_state(
    mock_bjobs, tmpdir, monkeypatch
):
    monkeypatch.chdir(tmpdir)

    lsf_driver = LSFDriver(queue_options=[("BJOBS_CMD", tmpdir / "mock_bjobs")])
    lsf_driver._max_attempt = 3
    lsf_driver._retry_sleep_period = 0
    create_fake_bjobs_result(tmpdir, job_id="valid_job_id", status="RUN")

    mock_realization_state = MockRealizationState()
    mock_realization_state.current_state.id = "PENDING"
    lsf_driver._realstate_to_lsfid[mock_realization_state] = "valid_job_id"
    lsf_driver._lsfid_to_realstate["valid_job_id"] = mock_realization_state

    await lsf_driver.poll_statuses()

    bjobs_logs = Path("bjobs_log").read_text(encoding="utf-8").strip().split("\n")

    # Should only call bjobs once
    assert len(bjobs_logs) == 1

    # Should update realization state
    assert mock_realization_state.current_state.id == "RUNNING"


@pytest.mark.asyncio
async def test_kill_bkill_non_existent_jobid_exit_code_1(
    mock_bkill, caplog, tmpdir, monkeypatch
):
    monkeypatch.chdir(tmpdir)
    lsf_driver = LSFDriver(queue_options=[("BKILL_CMD", tmpdir / "mock_bkill")])
    lsf_driver._max_attempt = 3
    lsf_driver._retry_sleep_period = 0
    mock_realization_state = MockRealizationState()
    lsf_driver._realstate_to_lsfid[mock_realization_state] = "non_existent_jobid"

    with pytest.raises(RuntimeError, match="Maximum number of kill errors exceeded"):
        await lsf_driver.kill(mock_realization_state)

    output = caplog.text
    job_ids_from_file = Path("job_ids").read_text(encoding="utf-8").strip().split("\n")
    assert len(job_ids_from_file) == lsf_driver._max_attempt
    assert (
        len(re.findall("bkill: jobid non_existent_jobid not found", output))
        == lsf_driver._max_attempt
    )

    assert (
        len(re.findall("returned non-zero exitcode: 1", output))
        == lsf_driver._max_attempt
    )


@pytest.mark.parametrize(
    "options, expected_list",
    [
        [[("LSF_QUEUE", "pytest_queue")], ["bsub", "-q", "pytest_queue"]],
        [[("BSUB_CMD", "/bin/mock/bsub")], ["/bin/mock/bsub"]],
    ],
)
def test_lsf_build_submit_cmd_adds_driver_options(
    options: list[Tuple[str, str]], expected_list
):
    lsf_driver = LSFDriver(options)
    submit_command_list = lsf_driver.build_submit_cmd()
    assert submit_command_list == expected_list


@pytest.mark.parametrize(
    "additional_parameters", [[["test0", "test2", "/home/test3.py"]], [[3, 2]], [[]]]
)
def test_lsf_build_submit_cmd_adds_additional_parameters(
    additional_parameters: list[str],
):
    lsf_driver = LSFDriver(None)
    submit_command_list = lsf_driver.build_submit_cmd(*additional_parameters)
    assert submit_command_list == ["bsub"] + additional_parameters


@pytest.mark.parametrize(
    "options, additional_parameters, expected_list",
    [
        [
            [("LSF_QUEUE", "pytest_queue")],
            ["test0", "test2", "/home/test3.py"],
            ["bsub", "-q", "pytest_queue", "test0", "test2", "/home/test3.py"],
        ],
        [
            [("LSF_QUEUE", "pytest_queue"), ("BSUB_CMD", "/bin/test_bsub")],
            ["test0", "test2", "/home/test3.py"],
            [
                "/bin/test_bsub",
                "-q",
                "pytest_queue",
                "test0",
                "test2",
                "/home/test3.py",
            ],
        ],
    ],
)
def test_lsf_build_submit_cmd_adds_additional_parameters_after_options(
    options: list[tuple[str, str]],
    additional_parameters: list[str],
    expected_list: list[str],
):
    lsf_driver = LSFDriver(options)
    submit_command_list = lsf_driver.build_submit_cmd(*additional_parameters)
    assert submit_command_list == expected_list


@pytest.mark.parametrize(
    "driver_options, expected_bsub_options",
    [[[("LSF_QUEUE", "test_queue")], ["-q test_queue"]]],
)
@pytest.mark.asyncio
async def test_lsf_submit_lsf_queue_option_is_added(
    driver_options: list[Tuple[str, str]],
    expected_bsub_options: list[str],
    mock_bsub,
    tmpdir,
    monkeypatch,
):
    monkeypatch.chdir(tmpdir)

    lsf_driver = LSFDriver(
        queue_options=[("BSUB_CMD", tmpdir / "mock_bsub"), *driver_options]
    )

    mock_realization_state = MockRealizationState()
    mock_realization_state.realization.job_script = "valid_script.sh"
    await lsf_driver.submit(mock_realization_state)

    command_called = Path("bsub_log").read_text(encoding="utf-8").strip()
    assert len(command_called.split("\n")) == 1

    for expected_bsub_option in expected_bsub_options:
        assert expected_bsub_option in command_called


@pytest.fixture
def copy_lsf_poly_case(copy_poly_case, tmp_path):
    # Overwriting the "poly.ert" config file in tmpdir runpath
    # with our own customized config with at least sets queue option to LSF and
    # introducing the mocked jobs.

    config = [
        "JOBNAME poly_%d\n",
        "QUEUE_SYSTEM LSF\n",
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


@pytest.fixture
def mock_bjobs(tmp_path):
    script = "#!/usr/bin/env python3" + dedent(
        """
        import datetime
        import json
        import os.path
        import sys
        import time

        # Just to have a log for test purposes what is actually thrown
        # towards the bjobs command
        with open("bjobs_log", "a+", encoding="utf-8") as f:
            f.write(f"{str(sys.argv)}\\n")


        print("JOBID USER STAT QUEUE FROM_HOST EXEC_HOST JOB_NAME SUBMIT_TIME")

        timestamp = str(datetime.datetime.now())
        # File written from the mocked bsub command which provides us with
        # the path to where the job actually runs and where we can find i.e
        # the job_id and status
        if not os.path.isfile("job_paths"):
            with open("mocked_result", mode="r+", encoding="utf-8") as result_line:
                result = result_line.read()
            if "exit" in result.split("\t"):
                exit(1)
            print(result)

        else:
            with open("job_paths", encoding="utf-8") as job_paths_file:
                job_paths = job_paths_file.read().splitlines()
            for line in job_paths:
                if not os.path.isfile(line + "/lsf_info.json"):
                    continue

                # ERT has picked up the mocked response from mock_bsub and
                # written the id to file
                with open(line + "/lsf_info.json") as id_file:
                   _id = json.load(id_file)["job_id"]

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
                status = "RUN"
                if os.path.isfile(f"{line}/OK"):
                    status = "DONE"

                # Together with the headerline this is actually how LSF is
                # providing its statuses on the job and how we are picking these
                # up. In this mocked version i just check if the job is done
                # with the OK file and then print that status for the job_id
                # retrieved from the same runpath.
                print(
                    f"{_id} pytest {status} normal"
                    f" mock_host mock_exec_host poly_0 {timestamp}"
                )
           """
    )
    script_path = tmp_path / "mock_bjobs"
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(script)

    os.chmod(script_path, 0o755)


def make_failing_bsub(script_path, success_script):
    """
    Approx 3/10 of the submits will fail due to the random generator in the
    created mocked bsub script. By using the retry functionality towards
    queue-errors in job_queue.cpp we should still manage to finalize all our runs
    before exhausting the limits
    """
    script_path.write_text(
        "#!/usr/bin/env python3"
        + dedent(
            f"""
            import random
            import sys
            import subprocess
            num = random.random()
            if num > 0.7:
                exit(1)
            subprocess.call(["python", "{success_script}"] + sys.argv)
            """
        )
    )

    os.chmod(script_path, 0o755)


def make_mock_bsub(script_path):
    script_path.write_text(
        "#!/usr/bin/env python3"
        + dedent(
            """
           import random
           import subprocess
           import sys
           import json
           from pathlib import Path
           job_dispatch_path = sys.argv[-2]
           run_path = sys.argv[-1]
           # Write a file with the runpaths to where the jobs are running and
           # writing information we later need when providing statuses for the
           # jobs through the mocked bjobs command
           with open("job_paths", "a+", encoding="utf-8") as jobs_file:
               jobs_file.write(f"{run_path}\\n")

           # Just a log for testpurposes showing what is thrown against the
           # bsub command
           with open("bsub_log", "a+", encoding="utf-8") as f:
               f.write(f"{str(sys.argv)}\\n")

           # Assigning a "unique" job id for each submitted job and print. This
           # is how LSF provide response to ERT with the ID of the job.
           _id = str(random.randint(0, 10000000))
           print(f"Job <{_id}> is submitted to default queue <normal>.")
           with open(Path(run_path,"lsf_info.json"), mode="a+", encoding="utf-8") as f:
                lsf_data = {
                    "job_id": _id
                }
                f.write(json.dumps(lsf_data))
           # Launch job-dispatch
           subprocess.Popen([job_dispatch_path, run_path])
           """
        )
    )
    os.chmod(script_path, 0o755)


@pytest.fixture(params=["success", "fail"])
def mock_bsub_integration_test(request, tmp_path):
    if request.param == "success":
        return make_mock_bsub(tmp_path / "mock_bsub")
    else:
        make_mock_bsub(tmp_path / "success_bsub")
        return make_failing_bsub(tmp_path / "mock_bsub", tmp_path / "success_bsub")


@pytest.mark.usefixtures(
    "copy_lsf_poly_case",
    "mock_bsub_integration_test",
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


@pytest.mark.skip(reason="Needs reimplementation")
@pytest.mark.usefixtures("use_tmpdir", "mock_fm_ok")
def test_num_cpu_submitted_correctly_lsf(tmpdir, simple_script):
    """Assert that num_cpu from the ERT configuration is passed on to the bsub
    command used to submit jobs to LSF"""
    os.putenv("PATH", os.getcwd() + ":" + os.getenv("PATH"))
    driver = Driver(driver_type=QueueSystem.LSF)

    bsub = Path("bsub")
    bsub.write_text(MOCK_BSUB, encoding="utf-8")
    bsub.chmod(stat.S_IRWXU)

    job_id = 0
    num_cpus = 4
    os.mkdir(DUMMY_CONFIG["run_path"].format(job_id))

    job = QueueableRealization(
        job_script=simple_script,
        num_cpu=4,
        run_arg=RunArg(
            str(job_id),
            MagicMock(spec=EnsembleAccessor),
            0,
            0,
            os.path.realpath(DUMMY_CONFIG["run_path"].format(job_id)),
            DUMMY_CONFIG["job_name"].format(job_id),
        ),
    )

    pool_sema = BoundedSemaphore(value=2)
    job.run(driver, pool_sema)
    job.stop()
    job.wait_for()

    bsub_argv: List[str] = Path("test.out").read_text(encoding="utf-8").split()

    found_cpu_arg = False
    for arg_i, arg in enumerate(bsub_argv):
        if arg == "-n":
            # Check that the driver submitted the correct number
            # of cpus:
            assert bsub_argv[arg_i + 1] == str(num_cpus)
            found_cpu_arg = True

    assert found_cpu_arg is True
