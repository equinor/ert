import json
from pathlib import Path
from typing import Any

info = "Add default values for weights, scales, objective_types to everest objectives"


def migrate_everest_objectives_config(
    everest_objectives_config: dict[str, Any],
) -> dict[str, Any]:
    num_objectives = len(everest_objectives_config["keys"])
    return everest_objectives_config | {
        "weights": [None] * num_objectives,
        "scales": [None] * num_objectives,
        "objective_types": ["mean"] * num_objectives,
    }


def migrate_everest_objectives_in_responses_json(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        responses_json = json.loads(
            (experiment / "responses.json").read_text(encoding="utf-8")
        )

        if "everest_objectives" in responses_json:
            responses_json["everest_objectives"] = migrate_everest_objectives_config(
                responses_json["everest_objectives"]
            )

            (experiment / "responses.json").write_text(
                json.dumps(responses_json, indent=2),
                encoding="utf-8",
            )


def migrate(path: Path) -> None:
    migrate_everest_objectives_in_responses_json(path)
