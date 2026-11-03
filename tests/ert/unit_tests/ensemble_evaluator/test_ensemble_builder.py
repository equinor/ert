from unittest.mock import MagicMock

import pytest

from ert.config import ForwardModelStep, QueueConfig, QueueSystem
from ert.config.ert_config import create_forward_model_json
from ert.config.forward_model_step import UserInstalledForwardModelStep
from ert.ensemble_evaluator._ensemble import LegacyEnsemble, Realization
from ert.ensemble_evaluator.state import FORWARD_MODEL_STATE_INIT


@pytest.mark.parametrize("active_real", [True, False])
def test_build_ensemble(active_real):
    ensemble = LegacyEnsemble(
        [
            Realization(
                iens=2,
                run_arg=MagicMock(),
                num_cpu=1,
                max_runtime=0,
                job_script="job_script",
                fm_steps=[ForwardModelStep(name="echo_command", executable="")],
                active=active_real,
                realization_memory=0,
            )
        ],
        {},
        QueueConfig(queue_system=QueueSystem.LOCAL),
        0,
        "1",
    )

    real = ensemble.reals[0]
    assert real.active == active_real
    if active_real:
        assert (
            ensemble.snapshot.get_fm_step("2", "0").get("status")
            == FORWARD_MODEL_STATE_INIT
        )


@pytest.mark.parametrize(
    ("label", "template", "expected"),
    [
        ("custom label", None, "custom label"),
        ("<PREFIX>_<INPUT>_<IENS>_<ITER>", None, "copy_input_2_3"),
        (None, "<PREFIX>_<INPUT>_<OUTPUT>_<IENS>_<ITER>", "copy_input_out_2_3"),
        ("override <INPUT>", "ignored", "override input"),
        (None, "<NESTED>", "<INPUT>"),
    ],
)
def test_that_snapshot_uses_forward_model_display_name(label, template, expected):
    step = UserInstalledForwardModelStep(
        name="echo_command",
        label=label,
        display_name_template=template,
        executable="",
        private_args={"<INPUT>": "input", "<NESTED>": "<INPUT>"},
        default_mapping={"<INPUT>": "ignored", "<OUTPUT>": "out"},
    )
    substitutions = {"<PREFIX>": "copy"}
    ensemble = LegacyEnsemble(
        [
            Realization(
                iens=2,
                run_arg=MagicMock(itr=3),
                num_cpu=1,
                max_runtime=0,
                job_script="job_script",
                fm_steps=[step],
                active=True,
                realization_memory=0,
            )
        ],
        {},
        QueueConfig(queue_system=QueueSystem.LOCAL),
        0,
        "1",
        substitutions=substitutions,
    )

    job = create_forward_model_json(
        context=substitutions, forward_model_steps=[step], run_id=None, iens=2, itr=3
    )["jobList"][0]
    assert (
        ensemble.snapshot.get_fm_step("2", "0").get("name") == job["name"] == expected
    )
