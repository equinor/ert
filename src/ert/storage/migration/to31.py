from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

info = "Strip fields not in the observation models from experiment observations"

_ALLOWED_OBSERVATION_KEYS: dict[str, frozenset[str]] = {
    "summary_observation": frozenset(
        {
            "type",
            "name",
            "value",
            "error",
            "key",
            "date",
            "shape_id",
            "error_mode",
            "error_min",
        }
    ),
    "general_observation": frozenset(
        {
            "type",
            "name",
            "data",
            "value",
            "error",
            "restart",
            "index",
            "shape_id",
        }
    ),
    "rft_observation": frozenset(
        {
            "type",
            "name",
            "well",
            "date",
            "property",
            "value",
            "error",
            "east",
            "north",
            "tvd",
            "md",
            "shape_id",
            "zone",
        }
    ),
    "breakthrough": frozenset(
        {
            "type",
            "name",
            "key",
            "date",
            "error",
            "threshold",
            "shape_id",
        }
    ),
}


def _strip_unknown_fields(path: Path) -> None:
    experiments_dir = path / "experiments"
    if not experiments_dir.exists():
        return

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        index_file = exp_dir / "index.json"
        if not index_file.exists():
            continue

        index_data = json.loads(index_file.read_text(encoding="utf-8"))
        experiment_data = index_data.get("experiment", {})
        observations = experiment_data.get("observations")
        if not observations:
            continue

        stripped_keys: set[str] = set()
        for observation in observations:
            allowed_keys = _ALLOWED_OBSERVATION_KEYS.get(observation.get("type"))
            if allowed_keys is None:
                logger.warning(
                    "Cannot migrate observation with unknown type "
                    f"{observation.get('type')} in {index_file}"
                )
                continue

            unknown_keys = [key for key in observation if key not in allowed_keys]
            for key in unknown_keys:
                observation.pop(key)
            stripped_keys.update(unknown_keys)

        if stripped_keys:
            logger.info(
                "Stripped fields not in the observation models from %s: %s",
                index_file,
                ", ".join(sorted(stripped_keys)),
            )
            index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")


def migrate(path: Path) -> None:
    _strip_unknown_fields(path)
