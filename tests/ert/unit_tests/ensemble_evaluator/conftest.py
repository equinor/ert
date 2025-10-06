import asyncio
import json
import os
import stat
from collections.abc import Callable
from contextlib import _AsyncGeneratorContextManager, asynccontextmanager
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, Mock

import pytest

import ert.ensemble_evaluator
from _ert.events import EEEvent
from ert.config import QueueConfig, QueueSystem
from ert.config.ert_config import forward_model_step_from_config_contents
from ert.config.queue_config import LocalQueueOptions
from ert.ensemble_evaluator._ensemble import LegacyEnsemble
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert.run_arg import RunArg
from ert.storage import Ensemble
from ert.storage.load_status import LoadResult
from tests.ert import SnapshotBuilder

from .ensemble_evaluator_utils import TestEnsemble


@pytest.fixture
def snapshot():
    return (
        SnapshotBuilder()
        .add_fm_step(
            fm_step_id="0",
            index="0",
            name="forward_model0",
            status="Unknown",
        )
        .add_fm_step(
            fm_step_id="1",
            index="1",
            name="forward_model1",
            status="Unknown",
        )
        .add_fm_step(
            fm_step_id="2",
            index="2",
            name="forward_model2",
            status="Unknown",
        )
        .add_fm_step(
            fm_step_id="3",
            index="3",
            name="forward_model3",
            status="Unknown",
        )
        # .build(["0","1"], status="Unknown")
        .build(["0", "1", "3", "4", "5", "9"], status="Unknown")
    )


@pytest.fixture(name="queue_config")
def queue_config_fixture():
    return QueueConfig(
        job_script="fm_dispatch.py",
        max_submit=1,
        queue_system=QueueSystem.LOCAL,
        queue_options=LocalQueueOptions(max_running=50),
    )


@pytest.fixture
def make_ensemble(queue_config):
    def _make_ensemble_builder(
        monkeypatch, tmpdir: Path, num_reals, num_jobs, job_sleep=0
    ):
        async def load_successful(**_):
            return LoadResult.success()

        monkeypatch.setattr(
            ert.scheduler.job,
            "load_realization_parameters_and_responses",
            load_successful,
        )
        with tmpdir.as_cwd():
            forward_model_list = []
            for job_index in range(num_jobs):
                forward_model_exec = Path(tmpdir) / f"ext_{job_index}.py"
                forward_model_exec.write_text(
                    "#!/usr/bin/env python\n"
                    "import time\n"
                    "\n"
                    'if __name__ == "__main__":\n'
                    f'    print("stdout from {job_index}")\n'
                    f"    time.sleep({job_sleep})\n"
                    f"    with open('status.txt', 'a', encoding='utf-8'): pass\n",
                    encoding="utf-8",
                )
                mode = os.stat(forward_model_exec).st_mode
                mode |= stat.S_IXUSR | stat.S_IXGRP
                os.chmod(forward_model_exec, stat.S_IMODE(mode))

                forward_model_list.append(
                    forward_model_step_from_config_contents(
                        f"EXECUTABLE ext_{job_index}.py\n",
                        str(Path(tmpdir) / f"EXT_JOB_{job_index}"),
                        name=f"forward_model_{job_index}",
                    )
                )
            realizations = []
            for iens in range(num_reals):
                run_path = Path(tmpdir / f"real_{iens}")
                run_path.mkdir()
                (run_path / "jobs.json").write_text(
                    json.dumps(
                        {
                            "jobList": [
                                _dump_forward_model(forward_model, index)
                                for index, forward_model in enumerate(
                                    forward_model_list
                                )
                            ],
                        },
                    ),
                    encoding="utf-8",
                )

                realizations.append(
                    ert.ensemble_evaluator.Realization(
                        active=True,
                        iens=iens,
                        fm_steps=forward_model_list,
                        job_script="fm_dispatch.py",
                        max_runtime=10,
                        num_cpu=1,
                        run_arg=RunArg(
                            str(iens),
                            MagicMock(spec=Ensemble),
                            iens,
                            0,
                            str(run_path),
                            f"job_name_{iens}",
                        ),
                        realization_memory=0,
                    )
                )

        ecl_config = Mock()
        ecl_config.assert_restart = Mock()

        return LegacyEnsemble(
            realizations,
            {},
            queue_config,
            0,
            "0",
        )

    return _make_ensemble_builder


def _dump_forward_model(forward_model, index):
    return {
        "name": forward_model.name,
        "executable": forward_model.executable,
        "target_file": forward_model.target_file,
        "error_file": forward_model.error_file,
        "start_file": forward_model.start_file,
        "stdout": f"{index}.stdout",
        "stderr": f"{index}.stderr",
        "stdin": forward_model.stdin_file,
        "environment": None,
        "max_running_minutes": forward_model.max_running_minutes,
        "min_arg": forward_model.min_arg,
        "max_arg": forward_model.max_arg,
        "arg_types": forward_model.arg_types,
        "argList": forward_model.arglist,
        "required_keywords:": forward_model.required_keywords,
    }


@pytest.fixture(name="make_ee_config")
def make_ee_config_fixture():
    def _ee_config(**kwargs):
        return EvaluatorServerConfig(**kwargs)

    return _ee_config


@pytest.fixture
def evaluator_to_use(
    make_ee_config,
) -> _AsyncGeneratorContextManager[EnsembleEvaluator, None]:
    @asynccontextmanager
    async def _evaluator_to_use(
        end_event: Event | None = None,
        event_handler: Callable[[EEEvent], None] | None = None,
        ensemble: TestEnsemble | LegacyEnsemble | None = None,
    ):
        if end_event is None:
            end_event = Event()
        if ensemble is None:
            ensemble = TestEnsemble(0, 2, 2, id_="0")
        evaluator = EnsembleEvaluator(
            ensemble, make_ee_config(use_token=False), end_event, event_handler
        )
        evaluator._batching_interval = 0.5  # batching can be faster for tests
        run_task = asyncio.create_task(evaluator.run_and_get_successful_realizations())
        await evaluator._server_started
        try:
            yield evaluator
        finally:
            evaluator.stop()
            await run_task

    return _evaluator_to_use
