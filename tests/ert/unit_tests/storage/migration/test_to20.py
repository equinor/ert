import json
from pathlib import Path

from ert.storage.migration.to20 import config_without_name_attr, migrate


def test_that_name_is_removed_from_responses_json_for_single_object():
    summary_config = {
        "name": "summary",
        "input_files": ["CASE"],
        "keys": ["FOPR"],
        "has_finalized_keys": True,
        "refcase": ["1996-01-03 00:00:00", "1996-01-02 00:00:00"],
        "type": "summary",
    }

    result = config_without_name_attr(summary_config)

    assert "name" not in result
    summary_config.pop("name")
    assert summary_config == result
    assert result == config_without_name_attr(result)


def test_that_name_is_removed_from_responses_json_file_for_all_experiments(use_tmpdir):
    gendata_with_name = {
        "name": "gen_data",
        "input_files": ["gen%d.txt"],
        "keys": ["GEN"],
        "has_finalized_keys": True,
        "report_steps_list": [[1]],
        "type": "gen_data",
    }

    summary_with_name = {
        "name": "summary",
        "input_files": ["CASE"],
        "keys": ["FOPR"],
        "has_finalized_keys": True,
        "refcase": ["1996-01-03 00:00:00", "1996-01-02 00:00:00"],
        "type": "summary",
    }

    gendata_without_name = {**gendata_with_name}
    gendata_without_name.pop("name")
    summary_without_name = {**summary_with_name}
    summary_without_name.pop("name")

    experiment_paths = []
    for i in range(5):
        experiment_path = Path("experiments") / f"exp_{i}"
        experiment_path.mkdir(parents=True, exist_ok=True)
        experiment_paths.append(experiment_path)

        # Add responses.json to each experiment
        with open(experiment_path / "responses.json", "w", encoding="utf-8") as f:
            json.dump(
                {"gen_data": gendata_with_name, "summary": summary_with_name},
                f,
                indent=2,
            )

    migrate(Path("."))

    # Validate responses.json files no longer contain 'name' field
    for experiment_path in experiment_paths:
        with open(experiment_path / "responses.json", encoding="utf-8") as f:
            migrated_responses = json.load(f)

        assert migrated_responses == {
            "gen_data": gendata_without_name,
            "summary": summary_without_name,
        }
