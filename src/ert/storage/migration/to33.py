from __future__ import annotations

import logging
from pathlib import Path

from .migration_utils import experiment_index_files, read_json, write_json

logger = logging.getLogger(__name__)

info = "Strip fields not in the response config models from experiment responses"

_ALLOWED_RESPONSE_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "summary": frozenset(
        {
            "type",
            "input_files",
            "keys",
            "has_finalized_keys",
        }
    ),
    "gen_data": frozenset(
        {
            "type",
            "input_files",
            "keys",
            "report_steps_list",
            "has_finalized_keys",
        }
    ),
    "rft": frozenset(
        {
            "type",
            "name",
            "has_finalized_keys",
            "input_files",
            "keys",
            "data_to_read",
            "zonemap",
            "approximate_missing_values",
        }
    ),
    "everest_constraints": frozenset(
        {
            "type",
            "has_finalized_keys",
            "input_files",
            "keys",
            "scales",
            "targets",
            "upper_bounds",
            "lower_bounds",
        }
    ),
    "everest_objectives": frozenset(
        {
            "type",
            "has_finalized_keys",
            "input_files",
            "keys",
            "scales",
            "weights",
            "objective_types",
        }
    ),
    "breakthrough": frozenset(
        {
            "type",
            "keys",
            "summary_keys",
            "thresholds",
            "observed_dates",
            "has_finalized_keys",
        }
    ),
}


def _strip_unknown_fields(path: Path) -> None:
    for index_file in experiment_index_files(path):
        index_data = read_json(index_file)
        experiment_data = index_data.get("experiment", {})

        response_lists = [
            experiment_data.get("response_configuration"),
            experiment_data.get("derived_response_configuration"),
        ]

        stripped_keys: set[str] = set()
        for response_list in response_lists:
            if not response_list:
                continue

            for response_config in response_list:
                allowed_keys = _ALLOWED_RESPONSE_CONFIG_KEYS.get(
                    response_config.get("type")
                )
                if allowed_keys is None:
                    logger.warning(
                        "Cannot migrate response config with unknown type "
                        f"{response_config.get('type')} in {index_file}"
                    )
                    continue

                unknown_keys = [
                    key for key in response_config if key not in allowed_keys
                ]
                for key in unknown_keys:
                    response_config.pop(key)
                stripped_keys.update(unknown_keys)

        if stripped_keys:
            logger.info(
                "Stripped fields not in the response config models from %s: %s",
                index_file,
                ", ".join(sorted(stripped_keys)),
            )
            write_json(index_file, index_data)


def migrate(path: Path) -> None:
    _strip_unknown_fields(path)
