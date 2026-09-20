import json
from pathlib import Path

from ert.storage.migration.to35 import migrate


def _write_index(exp_path: Path, experiment: dict) -> Path:
    exp_path.mkdir(parents=True)
    index_file = exp_path / "index.json"
    index_file.write_text(json.dumps({"experiment": experiment}), encoding="utf-8")
    return index_file


def test_that_migration_moves_weights_into_analysis_settings(tmp_path):
    root = tmp_path / "project"
    index_file = _write_index(
        root / "experiments" / "experiment_1", {"weights": "4, 2, 1"}
    )

    migrate(root)

    experiment = json.loads(index_file.read_text(encoding="utf-8"))["experiment"]
    assert "weights" not in experiment
    assert experiment["analysis_settings"] == {"weights": "4, 2, 1"}


def test_that_migration_keeps_existing_analysis_settings_weights(tmp_path):
    root = tmp_path / "project"
    index_file = _write_index(
        root / "experiments" / "experiment_1",
        {"weights": "4, 2, 1", "analysis_settings": {"weights": "1, 1"}},
    )

    migrate(root)

    experiment = json.loads(index_file.read_text(encoding="utf-8"))["experiment"]
    assert "weights" not in experiment
    assert experiment["analysis_settings"]["weights"] == "1, 1"


def test_that_migration_leaves_experiments_without_weights_untouched(tmp_path):
    root = tmp_path / "project"
    index_file = _write_index(
        root / "experiments" / "experiment_1",
        {"experiment_type": "Ensemble Experiment"},
    )

    migrate(root)

    experiment = json.loads(index_file.read_text(encoding="utf-8"))["experiment"]
    assert experiment == {"experiment_type": "Ensemble Experiment"}
