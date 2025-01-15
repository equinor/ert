import difflib
from os.path import exists
from pathlib import Path

from seba_sqlite.snapshot import SebaSnapshot

from everest.bin.config_branch_script import config_branch_entry
from everest.config_file_loader import load_yaml
from everest.config_keys import ConfigKeys as CK


def test_config_branch_entry(cached_example):
    path, _, _ = cached_example("math_func/config_advanced.yml")

    config_branch_entry(["config_advanced.yml", "new_restart_config.yml", "-b", "1"])

    assert exists("new_restart_config.yml")

    old_config = load_yaml("config_advanced.yml")
    old_controls = old_config[CK.CONTROLS]

    assert CK.INITIAL_GUESS in old_controls[0]

    new_config = load_yaml("new_restart_config.yml")
    new_controls = new_config[CK.CONTROLS]

    assert CK.INITIAL_GUESS not in new_controls[0]
    assert len(new_controls) == len(old_controls)
    assert len(new_controls[0][CK.VARIABLES]) == len(old_controls[0][CK.VARIABLES])

    opt_controls = {}

    snapshot = SebaSnapshot(Path(path) / "everest_output" / "optimization_output")
    for opt_data in snapshot._optimization_data():
        if opt_data.batch_id == 1:
            opt_controls = opt_data.controls

    new_controls_initial_guesses = {
        var[CK.INITIAL_GUESS] for var in new_controls[0][CK.VARIABLES]
    }
    opt_control_val_for_batch_id = {v for k, v in opt_controls.items()}

    assert new_controls_initial_guesses == opt_control_val_for_batch_id


def test_config_branch_preserves_config_section_order(cached_example):
    path, _, _ = cached_example("math_func/config_advanced.yml")

    config_branch_entry(["config_advanced.yml", "new_restart_config.yml", "-b", "1"])

    assert exists("new_restart_config.yml")

    opt_controls = {}

    snapshot = SebaSnapshot(Path(path) / "everest_output" / "optimization_output")
    for opt_data in snapshot._optimization_data():
        if opt_data.batch_id == 1:
            opt_controls = opt_data.controls

    opt_control_val_for_batch_id = {v for k, v in opt_controls.items()}

    diff_lines = []
    with (
        open("config_advanced.yml", encoding="utf-8") as initial_config,
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
    assert "-initial_guess:0.25" in diff_lines
    for control_val in opt_control_val_for_batch_id:
        assert f"+initial_guess:{control_val}" in diff_lines
