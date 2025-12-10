import json
from pathlib import Path
from typing import Any

info = "Migrate realization error type from IntEnum to StrEnum"


def migrate_realization_errors_json_content(
    error_json: dict[str, Any],
) -> dict[str, Any]:
    int_to_str = {
        1: "undefined",
        2: "parameters_loaded",
        4: "responses_loaded",
        8: "failure_in_current",
        16: "failure_in_parent",
        # To cover cases  who ran with storage version 16
        # and StrEnum _RealizationStorageState
        "undefined": "undefined",
        "parameters_loaded": "parameters_loaded",
        "responses_loaded": "responses_loaded",
        "failure_in_current": "failure_in_current",
        "failure_in_parent": "failure_in_parent",
    }
    return error_json | {"type": int_to_str[error_json["type"]]}


def migrate_realization_errors(path: Path) -> None:
    for realization_error in path.glob("ensembles/*/realization-*/error.json"):
        old_error_json_content = json.loads(
            realization_error.read_text(encoding="utf-8")
        )
        realization_error.write_text(
            json.dumps(
                migrate_realization_errors_json_content(old_error_json_content),
                indent=2,
            )
        )


def migrate(path: Path) -> None:
    migrate_realization_errors(path)
