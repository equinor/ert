import json
from pathlib import Path

from ert.storage.migration.to21 import config_without_refcase, migrate


def test_that_refcase_is_removed_from_responses_json_for_single_object():
    summary_config = {
        "input_files": ["CASE"],
        "keys": ["FOPR"],
        "has_finalized_keys": True,
        "refcase": ["1996-01-03 00:00:00", "1996-01-02 00:00:00"],
        "type": "summary",
    }

    result = config_without_refcase(summary_config)

    assert "refcase" not in result
    summary_config.pop("refcase")
    assert summary_config == result
    assert result == config_without_refcase(result)


def test_that_name_is_removed_from_responses_json_file_for_all_experiments(use_tmpdir):
    gendata = {
        "input_files": ["gen%d.txt"],
        "keys": ["GEN"],
        "has_finalized_keys": True,
        "report_steps_list": [[1]],
        "type": "gen_data",
    }

    summary_with_refcase = {
        "input_files": ["CASE"],
        "keys": ["FOPR"],
        "has_finalized_keys": True,
        "refcase": ["1996-01-03 00:00:00", "1996-01-02 00:00:00"],
        "type": "summary",
    }

    summary_without_refcase = {**summary_with_refcase}
    summary_without_refcase.pop("refcase")

    experiment_paths = []
    for i in range(5):
        experiment_path = Path("experiments") / f"exp_{i}"
        experiment_path.mkdir(parents=True, exist_ok=True)
        experiment_paths.append(experiment_path)

        # Add responses.json to each experiment
        with open(experiment_path / "responses.json", "w", encoding="utf-8") as f:
            json.dump(
                {"gen_data": gendata, "summary": summary_with_refcase},
                f,
                indent=2,
            )

    migrate(Path("."))

    # Validate responses.json files no longer contain 'name' field
    for experiment_path in experiment_paths:
        with open(experiment_path / "responses.json", encoding="utf-8") as f:
            migrated_responses = json.load(f)

        assert migrated_responses == {
            "gen_data": gendata,
            "summary": summary_without_refcase,
        }
