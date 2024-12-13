from pathlib import Path

import pytest

from everest.bin.config_branch_script import (
    _updated_initial_guess,
    opt_controls_by_batch,
)
from everest.config_file_loader import load_yaml
from everest.config_keys import ConfigKeys as CK


def test_get_controls_for_batch(cached_example):
    path, _, _ = cached_example("math_func/config_advanced.yml")

    assert (
        opt_controls_by_batch(Path(path) / "everest_output" / "optimization_output", 1)
        is not None
    )
    assert (
        opt_controls_by_batch(Path(path) / "everest_output" / "optimization_output", 42)
        is None
    )

    opt_controls = opt_controls_by_batch(
        Path(path) / "everest_output" / "optimization_output", 1
    )
    control_names = set(opt_controls.keys())
    expected_control_names = {"point_x-0", "point_x-1", "point_x-2"}

    assert control_names == expected_control_names

    expected_control_values = {
        0.172,
        0.258,
        0.139,
    }

    control_values = {v for k, v in opt_controls.items()}

    assert sorted(control_values) == pytest.approx(
        sorted(expected_control_values), rel=1e-2
    )


def test_update_controls_initial_guess(cached_example):
    path, _, _ = cached_example("math_func/config_advanced.yml")

    old_controls = load_yaml("config_advanced.yml")[CK.CONTROLS]

    assert len(old_controls) == 1

    old_ctrl = next(iter(old_controls), None)

    assert old_ctrl is not None
    assert CK.INITIAL_GUESS in old_ctrl

    for var in old_ctrl[CK.VARIABLES]:
        assert CK.INITIAL_GUESS not in var

    opt_controls = opt_controls_by_batch(
        Path(path) / "everest_output" / "optimization_output", 1
    )
    updated_controls = _updated_initial_guess(old_controls, opt_controls)
    updated_ctl = next(iter(updated_controls), None)

    assert len(updated_controls) == len(old_controls)
    assert updated_ctl is not None

    for var in updated_ctl[CK.VARIABLES]:
        assert CK.INITIAL_GUESS in var

    opt_ctrl_values = {v for k, v in opt_controls.items()}
    updated_initial_guesses = {
        var[CK.INITIAL_GUESS] for var in updated_ctl[CK.VARIABLES]
    }
    assert opt_ctrl_values == updated_initial_guesses
