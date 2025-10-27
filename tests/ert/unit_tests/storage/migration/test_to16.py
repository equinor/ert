import json
from pathlib import Path

from ert.storage.migration.to16 import (
    migrate_everest_objectives_config,
    migrate_everest_objectives_in_responses_json,
)


def test_that_defaults_are_added_to_single_objectives_config_object():
    original_config = {"keys": ["objective1", "objective2"]}

    expected_config = {
        "keys": ["objective1", "objective2"],
        "weights": [None, None],
        "scales": [None, None],
        "objective_types": ["mean", "mean"],
    }

    migrated_config = migrate_everest_objectives_config(original_config)
    assert migrated_config == expected_config


def test_that_defaults_are_added_to_objectives_for_multiple_experiments(
    use_tmpdir: Path,
):
    experiments_dir = Path.cwd() / "experiments"
    experiments_dir.mkdir()
    exp1_dir = experiments_dir / "exp1"
    exp1_dir.mkdir()
    exp2_dir = experiments_dir / "exp2"
    exp2_dir.mkdir()

    original_responses = {"everest_objectives": {"keys": ["objective1", "objective2"]}}
    (exp1_dir / "responses.json").write_text(
        json.dumps(original_responses, indent=2), encoding="utf-8"
    )
    (exp2_dir / "responses.json").write_text(
        json.dumps(original_responses, indent=2), encoding="utf-8"
    )

    migrate_everest_objectives_in_responses_json(Path.cwd())

    expected_responses = {
        "everest_objectives": {
            "keys": ["objective1", "objective2"],
            "weights": [None, None],
            "scales": [None, None],
            "objective_types": ["mean", "mean"],
        }
    }

    for experiment in [exp1_dir, exp2_dir]:
        updated_responses = json.loads(
            (experiment / "responses.json").read_text(encoding="utf-8")
        )
        assert updated_responses == expected_responses
