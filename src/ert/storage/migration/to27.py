from __future__ import annotations

from pathlib import Path
from typing import Any

from .migration_utils import experiment_index_files, read_json, write_json

DEFAULT_LOCALIZATION_RADIUS = 2000

info = "Migrate observation localization (east/north/radius) into ShapeRegistry"


def _register_shape(
    shapes: dict[int, dict[str, Any]],
    east: float,
    north: float,
    radius: float,
) -> int:

    for existing_id, existing_shape in shapes.items():
        if (
            existing_shape["east"] == east
            and existing_shape["north"] == north
            and existing_shape["radius"] == radius
        ):
            return existing_id

    new_id = max(shapes.keys(), default=-1) + 1
    shape = {
        "type": "circle",
        "east": east,
        "north": north,
        "radius": radius,
        "shape_id": new_id,
    }
    shapes[new_id] = shape
    return new_id


def _migrate_observations(path: Path) -> None:
    for index_file in experiment_index_files(path):
        index_data = read_json(index_file)
        experiment_data = index_data.get("experiment", {})
        observations = experiment_data.get("observations", [])

        shapes: dict[int, dict[str, Any]] = {}

        for obs in observations:
            obs_type = obs.get("type")

            if obs_type in {"summary_observation", "breakthrough"}:
                east = obs.get("east")
                north = obs.get("north")
                if east is not None and north is not None:
                    radius = obs.get("radius")
                    radius = (
                        radius if radius is not None else DEFAULT_LOCALIZATION_RADIUS
                    )
                    shape_id = _register_shape(shapes, east, north, radius)
                    obs["shape_id"] = shape_id
                    obs.pop("east", None)
                    obs.pop("north", None)
                    obs.pop("radius", None)
                else:
                    obs.setdefault("shape_id", None)

            elif obs_type == "rft_observation":
                east = obs.get("east")
                north = obs.get("north")
                if east is not None and north is not None:
                    radius = obs.pop("radius", None)
                    radius = (
                        radius if radius is not None else DEFAULT_LOCALIZATION_RADIUS
                    )
                    shape_id = _register_shape(shapes, east, north, radius)
                    obs["shape_id"] = shape_id
                else:
                    obs.setdefault("shape_id", None)

            elif obs_type == "general_observation":
                obs.setdefault("shape_id", None)

        experiment_data["shape_registry"] = {"shapes": shapes}
        write_json(index_file, index_data)


def migrate(path: Path) -> None:
    _migrate_observations(path)
