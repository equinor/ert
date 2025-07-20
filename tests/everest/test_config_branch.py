from pathlib import Path

import pytest

from everest.bin.config_branch_script import (
    _updated_initial_guess,
    opt_controls_by_batch,
)
from everest.config import EverestConfig
from everest.config_file_loader import load_yaml


@pytest.mark.integration_test
@pytest.mark.xdist_group("math_func/config_advanced.yml")
def test_get_controls_for_batch(cached_example):
    config_dir, config_file, _, _ = cached_example("math_func/config_advanced.yml")

    config = EverestConfig.load_file(Path(config_dir) / config_file)

    assert opt_controls_by_batch(config.storage_dir, 1) is not None
    assert opt_controls_by_batch(config.storage_dir, 42) is None

    opt_controls = opt_controls_by_batch(config.storage_dir, 1)
    control_names = set(opt_controls.keys())
    expected_control_names = {"point.x.0", "point.x.1", "point.x.2"}

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


@pytest.mark.integration_test
@pytest.mark.xdist_group("math_func/config_advanced.yml")
def test_update_controls_initial_guess(cached_example):
    config_path, config_file, _, _ = cached_example("math_func/config_advanced.yml")

    old_controls = load_yaml("config_advanced.yml")["controls"]

    assert len(old_controls) == 1

    old_ctrl = next(iter(old_controls), None)

    assert old_ctrl is not None
    assert "initial_guess" in old_ctrl

    for var in old_ctrl["variables"]:
        assert "initial_guess" not in var

    config = EverestConfig.load_file(Path(config_path) / config_file)
    opt_controls = opt_controls_by_batch(config.storage_dir, 1)
    updated_controls = _updated_initial_guess(old_controls, opt_controls)
    updated_ctl = next(iter(updated_controls), None)

    assert len(updated_controls) == len(old_controls)
    assert updated_ctl is not None

    for var in updated_ctl["variables"]:
        assert "initial_guess" in var

    opt_ctrl_values = {v for k, v in opt_controls.items()}
    updated_initial_guesses = {var["initial_guess"] for var in updated_ctl["variables"]}
    assert opt_ctrl_values == updated_initial_guesses
