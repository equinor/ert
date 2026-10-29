import json

from ert.storage.migration.to43 import migrate


def test_that_experiment_status_key_is_renamed_to_state(tmp_path):
    index_path = tmp_path / "experiments" / "exp-1" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "status": {
                    "message": "Experiment started",
                    "status": "running",
                }
            }
        ),
        encoding="utf-8",
    )

    migrate(tmp_path)

    assert json.loads(index_path.read_text(encoding="utf-8")) == {
        "status": {
            "message": "Experiment started",
            "state": "running",
        }
    }


def test_that_experiment_without_status_is_left_unchanged(tmp_path):
    index_path = tmp_path / "experiments" / "exp-1" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(json.dumps({"name": "experiment"}), encoding="utf-8")

    migrate(tmp_path)

    assert json.loads(index_path.read_text(encoding="utf-8")) == {"name": "experiment"}
