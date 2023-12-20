import asyncio
import json
import shutil
from pathlib import Path

import pytest

from ert.ensemble_evaluator._builder._realization import Realization
from ert.job_queue.queue import EVTYPE_ENSEMBLE_STOPPED
from ert.run_arg import RunArg
from ert.scheduler import scheduler


def create_jobs_json(realization: Realization) -> None:
    jobs = {
        "global_environment": {},
        "config_path": "/dev/null",
        "config_file": "/dev/null",
        "jobList": [
            {
                "name": forward_model.name,
                "executable": forward_model.executable,
                "argList": forward_model.arglist,
            }
            for forward_model in realization.forward_models
        ],
        "run_id": "0",
        "ert_pid": "0",
        "real_id": str(realization.iens),
    }
    realization_run_path = Path(realization.run_arg.runpath)
    realization_run_path.mkdir()
    with open(realization_run_path / "jobs.json", mode="w", encoding="utf-8") as f:
        json.dump(jobs, f)


@pytest.fixture
def realization(storage, tmp_path):
    ensemble = storage.create_experiment().create_ensemble(name="foo", ensemble_size=1)
    return create_stub_realization(ensemble, tmp_path, 0)


def create_stub_realization(ensemble, base_path: Path, iens) -> Realization:
    run_arg = RunArg(
        run_id="",
        ensemble_storage=ensemble,
        iens=iens,
        itr=0,
        runpath=str(base_path / f"realization-{iens}"),
        job_name="",
    )

    realization = Realization(
        iens=iens,
        forward_models=[],
        active=True,
        max_runtime=None,
        run_arg=run_arg,
        num_cpu=1,
        job_script=str(shutil.which("job_dispatch.py")),
    )
    return realization


async def test_empty():
    sch = scheduler.Scheduler()
    assert await sch.execute() == EVTYPE_ENSEMBLE_STOPPED


async def test_single_job(realization, mock_driver):
    future = asyncio.Future()

    async def init(iens, *args, **kwargs):
        future.set_result(iens)

    driver = mock_driver(init=init)

    sch = scheduler.Scheduler(driver)
    sch.add_realization(realization)

    assert await sch.execute() == EVTYPE_ENSEMBLE_STOPPED
    assert await future == realization.iens


async def test_cancel(realization, mock_driver):
    pre = asyncio.Event()
    post = asyncio.Event()
    killed = False

    async def wait():
        pre.set()
        await asyncio.sleep(10)
        post.set()

    async def kill():
        nonlocal killed
        killed = True

    driver = mock_driver(wait=wait, kill=kill)
    sch = scheduler.Scheduler(driver)
    sch.add_realization(realization)

    scheduler_task = asyncio.create_task(sch.execute())

    # Wait for the job to start
    await asyncio.wait_for(pre.wait(), timeout=1)

    # Kill all jobs and wait for the scheduler to complete
    sch.kill_all_jobs()
    await scheduler_task

    assert pre.is_set()
    assert not post.is_set()
    assert killed


async def test_add_dispatch_information_to_jobs_file(storage, tmp_path: Path):
    test_ee_uri = "ws://test_ee_uri.com/121/"
    test_ens_id = "test_ens_id121"
    test_ee_token = "test_ee_token_t0k€n121"
    test_ee_cert = "test_ee_cert121.pem"

    ensemble_size = 10

    ensemble = storage.create_experiment().create_ensemble(
        name="foo", ensemble_size=ensemble_size
    )
    realizations = [
        create_stub_realization(ensemble, tmp_path, iens)
        for iens in range(ensemble_size)
    ]

    sch = scheduler.Scheduler()
    sch.set_ee_info(test_ee_uri, test_ens_id, test_ee_cert, test_ee_token)

    for realization in realizations:
        sch.add_realization(realization)
        create_jobs_json(realization)

    sch.add_dispatch_information_to_jobs_file()

    for realization in realizations:
        job_file_path = Path(realization.run_arg.runpath, "jobs.json")
        content: dict = json.loads(job_file_path.read_text(encoding="utf-8"))
        assert content["ens_id"] == test_ens_id
        assert content["real_id"] == str(realization.iens)
        assert content["dispatch_url"] == test_ee_uri
        assert content["ee_token"] == test_ee_token
        assert content["ee_cert_path"] == test_ee_cert
        assert type(content["jobList"]) == list and len(content["jobList"]) == 0


@pytest.mark.parametrize(
    "max_submit",
    [
        (1),
        (2),
        (3),
    ],
)
async def test_that_max_submit_was_reached(realization, max_submit, mock_driver):
    retries = 0

    async def init(*args, **kwargs):
        nonlocal retries
        retries += 1

    async def wait():
        return False

    driver = mock_driver(init=init, wait=wait)
    sch = scheduler.Scheduler(driver)

    sch._max_submit = max_submit
    sch.add_realization(realization, callback_timeout=lambda _: None)

    assert await sch.execute() == EVTYPE_ENSEMBLE_STOPPED
    assert retries == max_submit
