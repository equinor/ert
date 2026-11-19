from __future__ import annotations

import logging
from pathlib import Path

from .migration_utils import experiment_index_files, read_json, write_json

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
    for index_file in experiment_index_files(path):
        index_data = read_json(index_file)
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
            write_json(index_file, index_data)


def migrate(path: Path) -> None:
    _strip_unknown_fields(path)
