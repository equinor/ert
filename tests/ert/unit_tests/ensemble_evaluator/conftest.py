import json
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

import ert.ensemble_evaluator
from ert.config import QueueConfig, QueueSystem
from ert.config.ert_config import _forward_model_step_from_config_file
from ert.config.queue_config import LocalQueueOptions
from ert.ensemble_evaluator._ensemble import LegacyEnsemble
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.load_status import LoadStatus
from ert.run_arg import RunArg
from ert.storage import Ensemble
from tests.ert import SnapshotBuilder


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
    def _make_ensemble_builder(monkeypatch, tmpdir, num_reals, num_jobs, job_sleep=0):
        async def load_successful(**_):
            return (LoadStatus.LOAD_SUCCESSFUL, "")

        monkeypatch.setattr(ert.scheduler.job, "forward_model_ok", load_successful)
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
                    _forward_model_step_from_config_file(
                        str(forward_model_config), name=f"forward_model_{job_index}"
                    )
                )
            realizations = []
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
        "exec_env": {},
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
        return EvaluatorServerConfig(custom_port_range=range(1024, 65535), **kwargs)

    return _ee_config
