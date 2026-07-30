import warnings
from pathlib import Path
from typing import Literal, TypedDict

from natsort import natsorted
from ruamel.yaml import YAML

from ert.cli.main import ErtCliError
from ert.config import ConfigValidationError, ErtConfig, Observation
from ert.plugins import ErtRuntimePlugins


class YamlObservation(TypedDict):
    date: str
    value: float
    error: float


class SummaryDict(TypedDict):
    key: str
    observations: list[YamlObservation]


YamlDict = dict[Literal["smry"], list[SummaryDict]]


class YamlConverter:
    TARGET_FILE = "summary_observations.yaml"

    def __init__(self, observations: list[Observation]) -> None:
        summary_observations = [
            o for o in observations if o.type == "summary_observation"
        ]
        if not summary_observations:
            raise ErtCliError("No summary observations in configuration.\nExiting ...")

        self.summary_observations = summary_observations

    def _summary_to_yaml_dict(self) -> YamlDict:
        summary_keys: set[str] = {o.key for o in self.summary_observations}
        summary_list: list[SummaryDict] = []
        for key in natsorted(summary_keys):
            observations_with_key = [
                o for o in self.summary_observations if o.key == key
            ]
            chronological_observations = sorted(
                observations_with_key, key=lambda o: o.date
            )
            # Round dates without HH/MM/SS to just date
            for o in chronological_observations:
                if o.date.endswith("T00:00:00"):
                    o.date = o.date.split("T")[0]
            obs_dicts: list[YamlObservation] = [
                {
                    "date": o.date,
                    "value": o.value,
                    "error": o.error,
                }
                for o in chronological_observations
            ]
            summary_dict: SummaryDict = {"key": key, "observations": obs_dicts}
            summary_list.append(summary_dict)
        return {"smry": summary_list}

    def export_yaml(self) -> None:
        yaml = YAML()
        yaml_dict = self._summary_to_yaml_dict()
        try:
            with Path(self.TARGET_FILE).open("x", encoding="utf-8") as f:
                yaml.dump(yaml_dict, f)
        except FileExistsError as error:
            raise ErtCliError(
                f"A file with name '{self.TARGET_FILE}' already exists. "
                "Will not overwrite it and exit instead."
            ) from error
        print(f"Successfully wrote summary observations to '{self.TARGET_FILE}'.")


def convert_summary_to_yaml(config: str, site_plugins: ErtRuntimePlugins) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore")
        try:
            ert_config = ErtConfig.with_plugins(site_plugins).from_file(config)
        except ConfigValidationError as e:
            raise ErtCliError(
                f"Failed to internalize the ert config '{config}' with error:\n    {e}"
            ) from e

    observations = ert_config.observation_declarations

    yaml_exporter = YamlConverter(
        observations=observations,
    )
    yaml_exporter.export_yaml()
