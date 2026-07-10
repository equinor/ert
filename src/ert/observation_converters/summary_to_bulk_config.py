import operator
from collections import defaultdict
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from natsort import natsorted

from ert.cli.main import ErtCliError
from ert.config import (
    ErtConfig,
    Observation,
    ShapeRegistry,
    SummaryKeyData,
    make_summary_key_data,
)
from ert.config._shapes import CircleShapeConfig

INDENT2 = " " * 2
INDENT4 = " " * 4
INDENT6 = " " * 6


def _shape_id_to_localization_str(
    obs_dicts: list[dict[str, Any]], shape_registry: ShapeRegistry
) -> str | None:
    shape_id = next(
        (o["shape_id"] for o in obs_dicts if o.get("shape_id") is not None), None
    )
    if shape_id is None:
        return None
    shape = shape_registry.get(shape_id)
    if shape is None or not isinstance(shape, CircleShapeConfig):
        return None
    lines = [
        f"{INDENT4}LOCALIZATION {{",
        f"{INDENT6}EAST={shape.east:g};",
        f"{INDENT6}NORTH={shape.north:g};",
    ]
    if shape.radius is not None:
        lines.append(f"{INDENT6}RADIUS={shape.radius:g};")
    lines.append(f"{INDENT4}}};")
    return "\n".join(lines)


def _non_empty_fields(skds: list[SummaryKeyData]) -> list[str]:
    if not skds:
        return []
    return [
        f.name
        for f in fields(SummaryKeyData)
        if any(getattr(o, f.name) is not None for o in skds)
    ]


def _breakthrough_to_string(obs: dict[str, Any], key: str) -> str:
    date = obs["date"]
    if isinstance(date, datetime):
        date = date.strftime("%Y-%m-%d")
    lines = [
        f"{INDENT4}BREAKTHROUGH {{",
        f"{INDENT6}THRESHOLD={obs['threshold']};",
        f"{INDENT6}DATE={date};",
        f"{INDENT6}ERROR={obs['error']};",
        f"{INDENT6}KEY={key};",
        f"{INDENT4}}};",
    ]
    return "\n".join(lines)


def _key_to_obs(
    observations: list[Observation],
    obs_type: Literal["summary_observation", "breakthrough"],
) -> dict[str, Any]:
    typed_obs = [obs for obs in observations if obs.type == obs_type]
    key_to_obs = defaultdict(list)
    for obs in typed_obs:
        key_to_obs[obs.key].append(dict(obs))
    return key_to_obs


class BulkConfigConverter:
    def __init__(
        self,
        observations: list[Observation],
        shape_registry: ShapeRegistry | None = None,
    ) -> None:
        self.shape_registry = shape_registry or ShapeRegistry()
        self.summary_observations = _key_to_obs(observations, "summary_observation")
        self.breakthrough_observations = _key_to_obs(observations, "breakthrough")
        self.csv_file_name = "summary_observations.csv"

        self.well_to_breakthrough = self._map_well_to_breakthrough()
        self.well_to_localization = self._map_localization_to_well()

        summary_keys = []
        for key in self.summary_observations:
            summary_key = make_summary_key_data(key)
            summary_keys.append(summary_key)

        self.header_fields = _non_empty_fields(summary_keys)

    def _map_localization_to_well(self) -> dict[str, Any]:
        well_to_localization = {}
        for key in self.summary_observations | self.breakthrough_observations:
            summary_key = make_summary_key_data(key.removeprefix("BREAKTHROUGH:"))
            loc_string = _shape_id_to_localization_str(
                self.summary_observations.get(key, [])
                + self.breakthrough_observations.get(key, []),
                self.shape_registry,
            )
            if loc_string is not None and (well := summary_key.well) is not None:
                well_to_localization[well] = loc_string
        return well_to_localization

    def _map_well_to_breakthrough(self) -> dict[str, Any]:
        well_to_breakthrough = {}
        for brt_key, obs in self.breakthrough_observations.items():
            key = brt_key.removeprefix("BREAKTHROUGH:")
            summary_key = make_summary_key_data(key)
            if (well := summary_key.well) is not None:
                if len(obs) > 1:
                    raise ErtCliError(
                        f"Can only have one breakthrough observation per well.\n"
                        f"Found {len(obs)} breakthroughs for well '{well}'."
                    )
                well_to_breakthrough[well] = _breakthrough_to_string(
                    obs[0], summary_key.keyword
                )
        return well_to_breakthrough

    def write_csv(self) -> None:
        with Path(self.csv_file_name).open(mode="w", encoding="utf-8") as fout:
            fout.write(", ".join([*self.header_fields, "value", "error", "date"]))
            fout.write("\n")
            # Sort observations chronologically before natsort
            for obs_list in self.summary_observations.values():
                obs_list.sort(key=operator.itemgetter("date"))
            for key in natsorted(self.summary_observations.keys()):
                skd = make_summary_key_data(key)
                for observation in self.summary_observations[key]:
                    for f in self.header_fields:
                        v = getattr(skd, f)
                        if v is None:
                            fout.write(", ")
                        else:
                            fout.write(f"{v}, ")
                    date = datetime.fromisoformat(observation["date"])
                    fout.write(f"{observation['value']:.3g}, ")
                    fout.write(f"{observation['error']:.3g}, ")
                    fout.write(f"{date.isoformat()}\n")

    def print_bulk_config(
        self,
    ) -> None:
        obs_names = [
            obs_dict["name"]
            for obs_list in self.summary_observations.values()
            for obs_dict in obs_list
        ] + [
            obs_dict["name"]
            for obs_list in self.breakthrough_observations.values()
            for obs_dict in obs_list
        ]
        obs_names_str = INDENT4 + f"\n{INDENT4}".join(obs_names)
        bulk_config_list = ["SUMMARY {", f"{INDENT2}VALUES = {self.csv_file_name};"]
        for well in sorted(
            self.well_to_localization.keys() | self.well_to_breakthrough.keys()
        ):
            bulk_config_list.extend(
                [
                    f"{INDENT2}WELL {well} {{",
                    self.well_to_localization.get(well, ""),
                    self.well_to_breakthrough.get(well, ""),
                    f"{INDENT2}}};",
                ]
            )
        bulk_config_list.append("};")
        bulk_config_str = "\n".join(
            filter(None, bulk_config_list)
            # Filter out empty (falsy) strings
        )
        print(
            f"\n{len(obs_names)} observations can be replaced by: \n"
            f"{INDENT2}1.  Copying the file '{self.csv_file_name}' to the folder "
            f"containing your observation configuration.\n"
            f"{INDENT2}2.  Replacing the named observations below with the bulk "
            f"configuration\n\n"
            f"Observation names (to replace):\n"
            f"==============================\n"
            f"{obs_names_str}\n\n"
            f"Bulk configuration (replace with):\n"
            f"=================================\n"
            f"{bulk_config_str}"
        )


def convert_summary_to_bulk_config(config: str) -> None:
    ert_config = ErtConfig.from_file(config)
    bulk_exporter = BulkConfigConverter(
        observations=ert_config.observation_declarations,
        shape_registry=ert_config.shape_registry,
    )
    bulk_exporter.write_csv()
    bulk_exporter.print_bulk_config()
