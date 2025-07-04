import json
from pathlib import Path

info = "Migrate response and parameter configs to include type"


def migrate(path: Path) -> None:
    def _replace_ert_kind(file: Path, kind_to_type: dict[str, str]) -> None:
        with open(file, encoding="utf-8") as fin:
            old_json = json.load(fin)
            new_json = {}

            for key, config in old_json.items():
                ert_kind = config.pop("_ert_kind")

                new_json[key] = config | {"type": kind_to_type[ert_kind]}

        with open(file, "w", encoding="utf-8") as fout:
            json.dump(new_json, fout, indent=2)

    for experiment in path.glob("experiments/*"):
        _replace_ert_kind(
            file=experiment / "parameter.json",
            kind_to_type={
                "GenKwConfig": "gen_kw",
                "ExtParamConfig": "everest_parameters",
                "Field": "field",
                "SurfaceConfig": "surface",
            },
        )

        _replace_ert_kind(
            file=experiment / "responses.json",
            kind_to_type={
                "GenDataConfig": "gen_data",
                "SummaryConfig": "summary",
                "EverestConstraintsConfig": "everest_constraints",
                "EverestObjectiveConfig": "everest_objective",
            },
        )

        with open(experiment / "responses.json", encoding="utf-8") as fin:
            old_json = json.load(fin)
            new_json = {}

            for key, config in old_json.items():
                if config["type"] == "summary" and "refcase" in config:
                    config.pop("refcase")

                new_json[key] = config
