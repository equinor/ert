# test_migrate.py
import json
from datetime import datetime
from pathlib import Path

from ert.storage.migration.to17 import (
    migrate,
    migrate_realization_errors_json_content,
)


def test_add_experiment_status_to_index_json():
    input_json = {
        "type": 8,
        "message": "An error occurred.",
        "time": datetime.now().isoformat(),
    }

    expected_output = {
        "type": "failure_in_current",
        "message": "An error occurred.",
        "time": input_json["time"],
    }

    result = migrate_realization_errors_json_content(input_json)
    assert migrate_realization_errors_json_content(input_json) == expected_output, (
        f"Expected {expected_output}, got {result}"
    )


def test_that_realization_storage_state_is_converted_to_strenum_for_all_reals(
    use_tmpdir,
):
    ensembles = ["ensemble_1", "ensemble_2", "ensemble_3"]

    for ensemble in ensembles:
        ensemble_path = Path(f"ensembles/{ensemble}")
        ensemble_path.mkdir(parents=True, exist_ok=True)

        for realization, _error_int, _error_str in [
            (0, 1, "undefined"),
            (1, 2, "parameters_loaded"),
            (2, 4, "responses_loaded"),
            (3, 8, "failure_in_current"),
            (4, 16, "failure_in_parent"),
        ]:
            real_dir = ensemble_path / f"realization-{realization}"
            real_dir.mkdir()
            (real_dir / "error.json").write_text(
                json.dumps(
                    {
                        "type": _error_int,
                        "message": f"Realization {realization} in {ensemble} failed.",
                        "time": datetime.now().isoformat(),
                    }
                ),
                encoding="utf-8",
            )

    migrate(Path("."))

    for ensemble in ensembles:
        ensemble_path = Path(f"ensembles/{ensemble}")
        for realization, _error_int, _error_str in [
            (0, 1, "undefined"),
            (1, 2, "parameters_loaded"),
            (2, 4, "responses_loaded"),
            (3, 8, "failure_in_current"),
            (4, 16, "failure_in_parent"),
        ]:
            error_file = ensemble_path / f"realization-{realization}/error.json"
            contents = json.loads(error_file.read_text(encoding="utf-8"))
            assert contents["type"] == _error_str


def test_realizations_with_missing_error_json(use_tmpdir):
    ensembles = ["ensemble_1", "ensemble_2", "ensemble_3"]

    missing_files = {
        "ensemble_1": [1, 3],
        "ensemble_2": [0, 4],
        "ensemble_3": [2, 3],
    }

    for ensemble in ensembles:
        ensemble_path = Path(f"ensembles/{ensemble}")
        ensemble_path.mkdir(parents=True, exist_ok=True)

        for realization, _error_int, _error_str in [
            (0, 1, "undefined"),
            (1, 2, "parameters_loaded"),
            (2, 4, "responses_loaded"),
            (3, 8, "failure_in_current"),
            (4, 16, "failure_in_parent"),
        ]:
            real_dir = ensemble_path / f"realization-{realization}"
            real_dir.mkdir()
            error_file = real_dir / "error.json"
            error_file.write_text(
                json.dumps(
                    {
                        "type": _error_int,
                        "message": f"Realization {realization} in {ensemble} failed.",
                        "time": datetime.now().isoformat(),
                    }
                ),
                encoding="utf-8",
            )

        for missing_real in missing_files[ensemble]:
            missing_file = ensemble_path / f"realization-{missing_real}/error.json"
            if missing_file.exists():
                missing_file.unlink()

    migrate(Path("."))

    for ensemble in ensembles:
        ensemble_path = Path(f"ensembles/{ensemble}")
        for realization, _error_int, _error_str in [
            (0, 1, "undefined"),
            (1, 2, "parameters_loaded"),
            (2, 4, "responses_loaded"),
            (3, 8, "failure_in_current"),
            (4, 16, "failure_in_parent"),
        ]:
            error_file = ensemble_path / f"realization-{realization}/error.json"

            if realization in missing_files[ensemble]:
                assert not error_file.exists()
            else:
                contents = json.loads(error_file.read_text(encoding="utf-8"))
                assert contents["type"] == _error_str
