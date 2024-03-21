import json
from pathlib import Path


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        responses_file = experiment / "responses.json"
        if path.exists():
            with open(responses_file, encoding="utf-8", mode="r") as f:
                info = json.load(f)
            for key, values in list(info.items()):
                if values.get("_ert_kind") == "SummaryConfig" and not values.get(
                    "keys"
                ):
                    del info[key]
            with open(responses_file, encoding="utf-8", mode="w") as f:
                json.dump(info, f)
