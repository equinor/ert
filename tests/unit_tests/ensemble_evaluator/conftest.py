import json
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

import ert.ensemble_evaluator
from ert.config import ForwardModel, QueueConfig, QueueSystem
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert.ensemble_evaluator.snapshot import SnapshotBuilder
from ert.job_queue import JobQueue
from ert.load_status import LoadStatus
from ert.run_arg import RunArg
from ert.storage import EnsembleAccessor

from .ensemble_evaluator_utils import TestEnsemble


@pytest.fixture
def snapshot():
    return (
        SnapshotBuilder()
        .add_job(
            job_id="0",
            index="0",
            name="job0",
            status="Unknown",
        )
        .add_job(
            job_id="1",
            index="1",
            name="job1",
            status="Unknown",
        )
        .add_job(
            job_id="2",
            index="2",
            name="job2",
            status="Unknown",
        )
        .add_job(
            job_id="3",
            index="3",
            name="job3",
            status="Unknown",
        )
        .build(["0", "1", "3", "4", "5", "9"], status="Unknown")
    )


@pytest.fixture(name="queue_config")
def queue_config_fixture():
    return QueueConfig(
        job_script="job_dispatch.py",
        max_submit=100,
        queue_system=QueueSystem.LOCAL,
        queue_options={QueueSystem.LOCAL: [("MAX_RUNNING", "50")]},
    )


@pytest.fixture
def make_ensemble_builder(queue_config):
    def _make_ensemble_builder(monkeypatch, tmpdir, num_reals, num_jobs, job_sleep=0):
        monkeypatch.setattr(
            "ert.job_queue.queue.forward_model_ok",
            lambda _: (LoadStatus.LOAD_SUCCESSFUL, ""),
        )
        builder = ert.ensemble_evaluator.EnsembleBuilder()
        with tmpdir.as_cwd():
            forward_model_list = []
            for job_index in range(0, num_jobs):
                forward_model_config = Path(tmpdir) / f"EXT_JOB_{job_index}"
                with open(forward_model_config, "w", encoding="utf-8") as f:
                    f.write(f"EXECUTABLE ext_{job_index}.py\n")

                forward_model_exec = Path(tmpdir) / f"ext_{job_index}.py"
                with open(forward_model_exec, "w", encoding="utf-8") as f:
                    f.write(
                        "#!/usr/bin/env python\n"
                        "import time\n"
                        "\n"
                        'if __name__ == "__main__":\n'
                        f'    print("stdout from {job_index}")\n'
                        f"    time.sleep({job_sleep})\n"
                        f"    with open('status.txt', 'a', encoding='utf-8'): pass\n"
                    )
                mode = os.stat(forward_model_exec).st_mode
                mode |= stat.S_IXUSR | stat.S_IXGRP
                os.chmod(forward_model_exec, stat.S_IMODE(mode))

                forward_model_list.append(
                    ForwardModel.from_config_file(
                        str(forward_model_config), name=f"forward_model_{job_index}"
                    )
                )

            for iens in range(0, num_reals):
                run_path = Path(tmpdir / f"real_{iens}")
                os.mkdir(run_path)

                with open(run_path / "jobs.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "jobList": [
                                _dump_forward_model(forward_model, index)
                                for index, forward_model in enumerate(
                                    forward_model_list
                                )
                            ],
                        },
                        f,
                    )

                builder.add_realization(
                    ert.ensemble_evaluator.RealizationBuilder()
                    .active(True)
                    .set_iens(iens)
                    .set_forward_models(forward_model_list)
                    .set_job_script("job_dispatch.py")
                    .set_max_runtime(10)
                    .set_num_cpu(1)
                    .set_run_arg(
                        RunArg(
                            str(iens),
                            MagicMock(spec=EnsembleAccessor),
                            iens,
                            0,
                            str(run_path),
                            f"job_name_{iens}",
                        ),
                    )
                )

        ecl_config = Mock()
        ecl_config.assert_restart = Mock()

        builder.set_legacy_dependencies(queue_config, False, 0)
        builder.set_id("0")
        return builder

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
        "exec_env": {},
        "max_running_minutes": forward_model.max_running_minutes,
        "min_arg": forward_model.min_arg,
        "max_arg": forward_model.max_arg,
        "arg_types": forward_model.arg_types,
        "argList": forward_model.arglist,
    }


@pytest.fixture(name="make_ee_config")
def make_ee_config_fixture():
    def _ee_config(**kwargs):
        return EvaluatorServerConfig(custom_port_range=range(1024, 65535), **kwargs)

    return _ee_config


@pytest.fixture
def evaluator(make_ee_config):
    ensemble = TestEnsemble(0, 2, 2, id_="0")
    ee = EnsembleEvaluator(
        ensemble,
        make_ee_config(),
        0,
    )
    yield ee
    ee.stop()
