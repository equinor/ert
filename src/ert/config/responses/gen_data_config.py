import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from typing_extensions import Self

from ert.config._option_dict import option_dict
from ert.config.parsing import ConfigValidationError, ErrorInfo
from ert.config.parsing.config_errors import ConfigWarning
from ert.config.parsing.observations_parser import GenObsValues, ObservationConfigError
from ert.config.responses import obs_commons
from ert.config.responses.general_observation import GenObservation
from ert.config.responses.observation_vector import ObsVector
from ert.config.responses.response_config import ObsArgs, ResponseConfig
from ert.config.responses.response_properties import (
    ResponseDataInitialLayout,
    ResponseTypes,
)
from ert.validation import rangestring_to_list


@dataclass
class GenDataConfig(ResponseConfig):
    @property
    def primary_keys(self) -> List[str]:
        return ["index", "report_step"]

    @property
    def response_type(self) -> str:
        return ResponseTypes.GEN_DATA

    @property
    def data_layout(self) -> ResponseDataInitialLayout:
        return ResponseDataInitialLayout.ONE_FILE_PER_NAME

    @staticmethod
    def _create_gen_obs(
        scalar_value: Optional[Tuple[float, float]] = None,
        obs_file: Optional[str] = None,
        data_index: Optional[str] = None,
    ) -> GenObservation:
        if scalar_value is None and obs_file is None:
            raise ValueError(
                "Exactly one the scalar_value and obs_file arguments must be present"
            )

        if scalar_value is not None and obs_file is not None:
            raise ValueError(
                "Exactly one the scalar_value and obs_file arguments must be present"
            )

        if obs_file is not None:
            try:
                file_values = np.loadtxt(obs_file, delimiter=None).ravel()
            except ValueError as err:
                raise ObservationConfigError.with_context(
                    f"Failed to read OBS_FILE {obs_file}: {err}", obs_file
                ) from err
            if len(file_values) % 2 != 0:
                raise ObservationConfigError.with_context(
                    "Expected even number of values in GENERAL_OBSERVATION", obs_file
                )
            values = file_values[::2]
            stds = file_values[1::2]

        else:
            assert scalar_value is not None
            obs_value, obs_std = scalar_value
            values = np.array([obs_value])
            stds = np.array([obs_std])

        if data_index is not None:
            indices = np.array([])
            if os.path.isfile(data_index):
                indices = np.loadtxt(data_index, delimiter=None, dtype=int).ravel()
            else:
                indices = np.array(
                    sorted(rangestring_to_list(data_index)), dtype=np.int32
                )
        else:
            indices = np.arange(len(values))
        std_scaling = np.full(len(values), 1.0)
        if len({len(stds), len(values), len(indices)}) != 1:
            raise ObservationConfigError.with_context(
                f"Values ({values}), error ({stds}) and "
                f"index list ({indices}) must be of equal length",
                obs_file if obs_file is not None else "",
            )
        return GenObservation(values, stds, indices, std_scaling)

    @staticmethod
    def parse_observation(args: ObsArgs) -> Dict[str, ObsVector]:
        general_observation = args.values
        assert type(general_observation) is GenObsValues
        assert general_observation is not None
        obs_key = args.obs_name
        time_map = args.obs_time_list
        has_refcase = args.refcase is not None
        config_node = args.config_for_response

        state_kw = general_observation.data
        if not config_node:
            ConfigWarning.ert_context_warn(
                f"Ensemble key {state_kw} does not exist"
                f" - ignoring observation {obs_key}",
                state_kw,
            )
            return {}

        if all(
            getattr(general_observation, key) is None
            for key in ["restart", "date", "days", "hours"]
        ):
            # The user has not provided RESTART or DATE, this is legal
            # for GEN_DATA, so we default it to None
            restart = None
        else:
            restart = obs_commons.get_restart(
                general_observation, obs_key, time_map, has_refcase
            )

        if not isinstance(config_node, GenDataConfig):
            ConfigWarning.ert_context_warn(
                f"{state_kw} has implementation type:"
                f"'{type(config_node)}' - "
                f"expected:'GEN_DATA' in observation:{obs_key}."
                "The observation will be ignored",
                obs_key,
            )
            return {}

        response_report_steps = (
            [] if config_node.report_steps is None else config_node.report_steps
        )
        if (restart is None and response_report_steps) or (
            restart is not None and restart not in response_report_steps
        ):
            ConfigWarning.ert_context_warn(
                f"The GEN_DATA node:{state_kw} is not configured to load from"
                f" report step:{restart} for the observation:{obs_key}"
                " - The observation will be ignored",
                state_kw,
            )
            return {}

        restart = 0 if restart is None else restart
        index_list = general_observation.index_list
        index_file = general_observation.index_file
        if index_list is not None and index_file is not None:
            raise ObservationConfigError.with_context(
                f"GENERAL_OBSERVATION {obs_key} has both INDEX_FILE and INDEX_LIST.",
                obs_key,
            )
        indices = index_list if index_list is not None else index_file
        try:
            return {
                obs_key: ObsVector(
                    ResponseTypes.GEN_DATA,
                    obs_key,
                    config_node.name,
                    {
                        restart: GenDataConfig._create_gen_obs(
                            (
                                (
                                    general_observation.value,
                                    general_observation.error,
                                )
                                if general_observation.value is not None
                                and general_observation.error is not None
                                else None
                            ),
                            general_observation.obs_file,
                            indices,
                        ),
                    },
                )
            }
        except ValueError as err:
            raise ObservationConfigError.with_context(str(err), obs_key) from err

    input_file: str = ""
    report_steps: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if isinstance(self.report_steps, list):
            self.report_steps = list(set(self.report_steps))

    @classmethod
    def from_config_list(cls, gen_data: List[str]) -> Self:
        options = option_dict(gen_data, 1)
        name = gen_data[0]
        res_file = options.get("RESULT_FILE")

        if res_file is None:
            raise ConfigValidationError.with_context(
                f"Missing or unsupported RESULT_FILE for GEN_DATA key {name!r}", name
            )

        report_steps: Optional[List[int]] = rangestring_to_list(
            options.get("REPORT_STEPS", "")
        )
        report_steps = sorted(list(report_steps)) if report_steps else None
        if os.path.isabs(res_file):
            result_file_context = next(
                x for x in gen_data if x.startswith("RESULT_FILE:")
            )
            raise ConfigValidationError.with_context(
                f"The RESULT_FILE:{res_file} setting for {name} is "
                f"invalid - must be a relative path",
                result_file_context,
            )

        if report_steps is None and "%d" in res_file:
            raise ConfigValidationError.from_info(
                ErrorInfo(
                    message="RESULT_FILES using %d must have REPORT_STEPS:xxxx"
                    " defined. Several report steps separated with ',' "
                    "and ranges with '-' can be listed",
                ).set_context_keyword(gen_data)
            )

        if report_steps is not None and "%d" not in res_file:
            result_file_context = next(
                x for x in gen_data if x.startswith("RESULT_FILE:")
            )
            raise ConfigValidationError.from_info(
                ErrorInfo(
                    message=f"When configuring REPORT_STEPS:{report_steps} "
                    "RESULT_FILES must be configured using %d"
                ).set_context_keyword(result_file_context)
            )
        return cls(name=name, input_file=res_file, report_steps=report_steps)

    def read_from_file(self, run_path: str, _: int) -> xr.Dataset:
        def _read_file(filename: Path, report_step: int) -> xr.Dataset:
            if not filename.exists():
                raise ValueError(f"Missing output file: {filename}")
            data = np.loadtxt(_run_path / filename, ndmin=1)
            active_information_file = _run_path / (str(filename) + "_active")
            if active_information_file.exists():
                active_list = np.loadtxt(active_information_file)
                data[active_list == 0] = np.nan
            return xr.Dataset(
                {"values": (["report_step", "index"], [data])},
                coords={
                    "index": np.arange(len(data)),
                    "report_step": [report_step],
                },
            )

        errors = []
        datasets = []
        filename_fmt = self.input_file
        _run_path = Path(run_path)
        if self.report_steps is None:
            try:
                datasets.append(_read_file(_run_path / filename_fmt, 0))
            except ValueError as err:
                errors.append(str(err))
        else:
            for report_step in self.report_steps:
                filename = filename_fmt % report_step  # noqa
                try:
                    datasets.append(_read_file(_run_path / filename, report_step))
                except ValueError as err:
                    errors.append(str(err))
        if errors:
            raise ValueError(f"Error reading GEN_DATA: {self.name}, errors: {errors}")
        return xr.combine_nested(datasets, concat_dim="report_step")
