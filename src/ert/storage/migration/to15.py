import json
from pathlib import Path

info = "Migrate metadata.json contents to be in experiment index.json .experiment field"


def migrate_metadata_json_into_experiment_index(path: Path) -> None:
    for experiment_path in path.glob("experiments/*"):
        experiment_json = {}
        with open(experiment_path / "metadata.json", encoding="utf-8") as fin:
            metadata_json = json.load(fin)
            if "weights" in metadata_json:
                experiment_json["weights"] = metadata_json["weights"]

        with open(experiment_path / "index.json", encoding="utf-8") as fin:
            index_json = json.load(fin)
            index_json["experiment"] = experiment_json

            Path(experiment_path / "index.json").write_text(
                json.dumps(index_json, indent=2), encoding="utf-8"
            )

        (experiment_path / "metadata.json").unlink()


def migrate(path: Path) -> None:
    migrate_metadata_json_into_experiment_index(path)
