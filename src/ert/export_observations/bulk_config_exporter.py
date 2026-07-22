from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeAlias

from natsort import natsorted

from ert.cli.main import ErtCliError
from ert.config import (
    SummaryKeyData,
    make_summary_key_data,
)

INDENT2 = " " * 2
INDENT4 = " " * 4
INDENT6 = " " * 6

LocalizationKeys = Literal["east", "north", "radius"]
LocalizationDict: TypeAlias = dict[LocalizationKeys, float | None]


def _localization_to_string(
    localization: LocalizationDict,
) -> str:
    if not localization:
        return ""
    lines = [
        f"{INDENT4}LOCALIZATION {{",
        f"{INDENT6}EAST={localization['east']};",
        f"{INDENT6}NORTH={localization['north']};",
    ]
    if (radius := localization["radius"]) is not None:
        lines.append(f"{INDENT6}RADIUS={radius};")
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
    lines = [
        f"{INDENT4}BREAKTHROUGH {{",
        f"{INDENT6}THRESHOLD={obs['values'][0]};",
        f"{INDENT6}DATE={obs['x_axis'][0]};",
        f"{INDENT6}ERROR={obs['errors'][0]};",
        f"{INDENT6}KEY={key};",
        f"{INDENT4}}};",
    ]
    return "\n".join(lines)


class BulkConfigExporter:
    def __init__(
        self,
        summary_observations: dict[str, Any],
        breakthrough_observations: dict[str, Any],
        well_localization: dict[str, LocalizationDict],
        csv_file_name: str = "summary_observation_values.csv",
    ) -> None:
        self.summary_observations = summary_observations
        self.breakthrough_observations = breakthrough_observations
        self.csv_file_name = csv_file_name
        self.well_to_localization = well_localization

        self.well_to_breakthrough = self._map_well_to_breakthrough()

        summary_keys = []
        for key in summary_observations:
            summary_key = make_summary_key_data(key)
            summary_keys.append(summary_key)

        self.header_fields = _non_empty_fields(summary_keys)

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
                obs_list.sort(key=lambda obs: obs["x_axis"][0])
            for key in natsorted(self.summary_observations.keys()):
                skd = make_summary_key_data(key)
                for observation in self.summary_observations[key]:
                    for f in self.header_fields:
                        v = getattr(skd, f)
                        if v is None:
                            fout.write(", ")
                        else:
                            fout.write(f"{v}, ")
                    date = datetime.fromisoformat(observation["x_axis"][0])
                    fout.write(f"{observation['values'][0]:.3g}, ")
                    fout.write(f"{observation['errors'][0]:.3g}, ")
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
                    _localization_to_string(self.well_to_localization.get(well, {})),
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
