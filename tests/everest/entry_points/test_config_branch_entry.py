import difflib
from os.path import exists
from pathlib import Path

import pytest

from everest.bin.config_branch_script import config_branch_entry
from everest.config import EverestConfig
from everest.config_file_loader import load_yaml
from everest.everest_storage import EverestStorage


@pytest.mark.integration_test
@pytest.mark.xdist_group("math_func/config_minimal.yml")
def test_config_branch_entry(cached_example):
    config_path, config_file, _, _ = cached_example("math_func/config_minimal.yml")

    config_branch_entry(["config_minimal.yml", "new_restart_config.yml", "-b", "1"])

    assert exists("new_restart_config.yml")

    old_config = load_yaml("config_minimal.yml")
    old_controls = old_config["controls"]

    assert "initial_guess" in old_controls[0]

    new_config = load_yaml("new_restart_config.yml")
    new_controls = new_config["controls"]

    assert "initial_guess" not in new_controls[0]
    assert len(new_controls) == len(old_controls)
    assert len(new_controls[0]["variables"]) == len(old_controls[0]["variables"])

    config = EverestConfig.load_file(Path(config_path) / config_file)
    storage = EverestStorage.from_storage_path(config.storage_dir)
    storage.read_from_output_dir()

    new_controls_initial_guesses = {
        var["initial_guess"] for var in new_controls[0]["variables"]
    }

    control_names = storage.controls["control_name"]
    batch_1_info = next(b for b in storage.batches if b.batch_id == 1)
    realization_control_vals = batch_1_info.realization_controls.select(
        *control_names
    ).to_dicts()[0]
    control_values = set(realization_control_vals.values())

    assert new_controls_initial_guesses == control_values


@pytest.mark.integration_test
@pytest.mark.xdist_group("math_func/config_minimal.yml")
def test_config_branch_preserves_config_section_order(cached_example):
    config_path, config_file, _, _ = cached_example("math_func/config_minimal.yml")

    config_branch_entry(["config_minimal.yml", "new_restart_config.yml", "-b", "1"])

    assert exists("new_restart_config.yml")

    diff_lines = []
    with (
        open("config_minimal.yml", encoding="utf-8") as initial_config,
        open("new_restart_config.yml", encoding="utf-8") as branch_config,
    ):
        diff = difflib.unified_diff(
            initial_config.readlines(),
            branch_config.readlines(),
            n=0,
        )
        for line in diff:
            if line.startswith("---"):
                continue
            if line.startswith("+++"):
                continue
            if line.startswith("@@"):
                continue
            diff_lines.append(line.replace(" ", "").strip())

    assert len(diff_lines) == 4
    assert "-initial_guess:0.1" in diff_lines

    config = EverestConfig.load_file(Path(config_path) / config_file)
    storage = EverestStorage.from_storage_path(config.storage_dir)
    storage.read_from_output_dir()
    control_names = storage.controls["control_name"]
    batch_1_info = next(b for b in storage.batches if b.batch_id == 1)
    realization_control_vals = batch_1_info.realization_controls.select(
        *control_names
    ).to_dicts()[0]
    control_values = set(realization_control_vals.values())

    for control_val in control_values:
        assert f"+initial_guess:{control_val}" in diff_lines
