import asyncio
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pytest

from ert.config.forward_model import ForwardModel
from ert.ensemble_evaluator._builder._realization import Realization
from ert.job_queue.queue import EVTYPE_ENSEMBLE_STOPPED
from ert.run_arg import RunArg
from ert.scheduler import scheduler


def create_bash_step(script: str) -> ForwardModel:
    return ForwardModel(
        name="bash_step",
        executable="/usr/bin/env",
        arglist=["bash", "-c", script],
    )


def create_jobs_json(path: Path, steps: Sequence[ForwardModel]) -> None:
    jobs = {
        "global_environment": {},
        "config_path": "/dev/null",
        "config_file": "/dev/null",
        "jobList": [
            {
                "name": step.name,
                "executable": step.executable,
                "argList": step.arglist,
            }
            for step in steps
        ],
        "run_id": "0",
        "ert_pid": "0",
        "real_id": "0",
    }

    with open(path / "jobs.json", "w") as f:
        json.dump(jobs, f)


@pytest.fixture
def realization(storage, tmp_path):
    ensemble = storage.create_experiment().create_ensemble(name="foo", ensemble_size=1)

    run_arg = RunArg(
        run_id="",
        ensemble_storage=ensemble,
        iens=0,
        itr=0,
        runpath=str(tmp_path),
        job_name="",
    )

    return Realization(
        iens=0,
        forward_models=[],
        active=True,
        max_runtime=None,
        run_arg=run_arg,
        num_cpu=1,
        job_script=str(shutil.which("job_dispatch.py")),
    )


async def test_empty():
    sch = scheduler.Scheduler()
    assert await sch.execute() == EVTYPE_ENSEMBLE_STOPPED


async def test_single_job(tmp_path: Path, realization):
    step = create_bash_step("echo 'Hello, world!' > testfile")
    realization.forward_models = [step]

    sch = scheduler.Scheduler()
    sch.add_realization(realization, callback_timeout=lambda _: None)

    create_jobs_json(tmp_path, [step])
    sch.add_dispatch_information_to_jobs_file()

    assert await sch.execute() == EVTYPE_ENSEMBLE_STOPPED
    assert (tmp_path / "testfile").read_text() == "Hello, world!\n"


async def test_cancel(tmp_path: Path, realization):
    step = create_bash_step("touch a; sleep 10; touch b")
    realization.forward_models = [step]

    sch = scheduler.Scheduler()
    sch.add_realization(realization, callback_timeout=lambda _: None)

    create_jobs_json(tmp_path, [step])
    sch.add_dispatch_information_to_jobs_file()

    scheduler_task = asyncio.create_task(sch.execute())

    # Wait for the job to start
    await asyncio.sleep(1)

    # Kill all jobs and wait for the scheduler to complete
    sch.kill_all_jobs()
    await scheduler_task

    assert (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()
