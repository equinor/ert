from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Set, Union

import numpy as np
import xarray as xr

from ert.config._read_summary import read_summary
from ert.config.parsing import ConfigWarning, HistorySource
from ert.config.parsing.observations_parser import (
    HistoryValues,
    ObservationConfigError,
    SummaryValues,
)
from ert.config.responses import obs_commons
from ert.config.responses.general_observation import GenObservation
from ert.config.responses.observation_vector import ObsVector
from ert.config.responses.response_config import ObsArgs, ResponseConfig
from ert.config.responses.response_properties import (
    ResponseDataInitialLayout,
    ResponseTypes,
)

from .summary_observation import SummaryObservation

if TYPE_CHECKING:
    from typing import List


logger = logging.getLogger(__name__)


def history_key(key: str) -> str:
    keyword, *rest = key.split(":")
    return ":".join([keyword + "H"] + rest)


@dataclass
class SummaryConfig(ResponseConfig):
    @staticmethod
    def _parse_summary_obs(args: ObsArgs) -> Dict[str, ObsVector]:
        summary_dict = args.values
        assert summary_dict is not None
        assert type(summary_dict) is SummaryValues
        summary_key = summary_dict.key
        time_map = args.obs_time_list
        value, std_dev = obs_commons.make_value_and_std_dev(summary_dict)
        obs_key = args.obs_name
        has_refcase = args.refcase is not None

        try:
            if summary_dict.date is not None and not time_map:
                # We special case when the user has provided date in SUMMARY_OBS
                # and not REFCASE or time_map so that we dont change current behavior.
                try:
                    date = datetime.fromisoformat(summary_dict.date)
                except ValueError as err:
                    raise ValueError("Please use ISO date format YYYY-MM-DD.") from err
                restart = None
            else:
                restart = obs_commons.get_restart(
                    summary_dict, obs_key, time_map, has_refcase
                )
                date = time_map[restart]
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Problem with date in summary observation {obs_key}: " + str(err),
                obs_key,
            ) from err

        if restart == 0:
            raise ObservationConfigError.with_context(
                "It is unfortunately not possible to use summary "
                "observations from the start of the simulation. "
                f"Problem with observation {obs_key}"
                f"{' at ' + str(obs_commons.get_time(summary_dict, time_map[0])) if summary_dict.restart is None else ''}",
                obs_key,
            )
        return {
            obs_key: ObsVector(
                ResponseTypes.SUMMARY,
                obs_key,
                summary_key,
                {date: SummaryObservation(summary_key, obs_key, value, std_dev)},
            )
        }

    @staticmethod
    def _parse_history_obs(args: ObsArgs) -> Dict[str, ObsVector]:
        refcase = args.refcase
        if refcase is None:
            raise ObservationConfigError("REFCASE is required for HISTORY_OBSERVATION")

        history_observation = args.values
        assert history_observation is not None

        history_type = args.history

        time_len = len(args.obs_time_list)
        obs_name = args.obs_name
        if history_type == HistorySource.REFCASE_HISTORY:
            local_key = history_key(args.obs_name)
        else:
            local_key = args.obs_name
        if local_key is None:
            return {}
        if local_key not in refcase.keys:
            return {}
        values = refcase.values[refcase.keys.index(local_key)]
        std_dev = obs_commons.handle_error_mode(values, history_observation)
        for segment_name, segment_instance in history_observation.segment:
            start = segment_instance.start
            stop = segment_instance.stop
            if start < 0:
                ConfigWarning.ert_context_warn(
                    f"Segment {segment_name} out of bounds."
                    " Truncating start of segment to 0.",
                    segment_name,
                )
                start = 0
            if stop >= time_len:
                ConfigWarning.ert_context_warn(
                    f"Segment {segment_name} out of bounds. Truncating"
                    f" end of segment to {time_len - 1}.",
                    segment_name,
                )
                stop = time_len - 1
            if start > stop:
                ConfigWarning.ert_context_warn(
                    f"Segment {segment_name} start after stop. Truncating"
                    f" end of segment to {start}.",
                    segment_name,
                )
                stop = start
            if np.size(std_dev[start:stop]) == 0:
                ConfigWarning.ert_context_warn(
                    f"Segment {segment_name} does not"
                    " contain any time steps. The interval "
                    f"[{start}, {stop}) does not intersect with steps in the"
                    "time map.",
                    segment_name,
                )
            std_dev[start:stop] = obs_commons.handle_error_mode(
                values[start:stop],
                segment_instance,
            )

        data: Dict[Union[int, datetime], Union[GenObservation, SummaryObservation]] = {}
        for i, (date, error, value) in enumerate(zip(refcase.dates, std_dev, values)):
            if error <= args.std_cutoff:
                ConfigWarning.ert_context_warn(
                    "Too small observation error in observation"
                    f" {obs_name}:{i} - ignored",
                    obs_name,
                )
                continue
                # Open question: Naming of history observations seems
                # a bit sketchy, not sure how to resolve
            data[date] = SummaryObservation(obs_name, obs_name, value, error)

        return {
            obs_name: ObsVector(
                ResponseTypes.SUMMARY,
                obs_name,
                obs_name,
                data,
            )
        }

    @staticmethod
    def parse_observation(args: ObsArgs) -> Dict[str, ObsVector]:
        if type(args.values) is HistoryValues:
            return SummaryConfig._parse_history_obs(args)

        elif type(args.values) == SummaryValues:
            return SummaryConfig._parse_summary_obs(args)

        raise KeyError(
            f"Expected history or summary observation, got {type(args.values)}"
        )

    @property
    def primary_keys(self) -> List[str]:
        return ["time"]

    @property
    def response_type(self) -> str:
        return ResponseTypes.SUMMARY

    @property
    def data_layout(self) -> ResponseDataInitialLayout:
        return ResponseDataInitialLayout.ONE_FILE_WITH_ALL_NAMES

    input_file: str
    keys: List[str]
    refcase: Union[Set[datetime], List[str], None] = None

    def __post_init__(self) -> None:
        if isinstance(self.refcase, list):
            self.refcase = {datetime.fromisoformat(val) for val in self.refcase}
        self.keys = sorted(set(self.keys))
        if len(self.keys) < 1:
            raise ValueError("SummaryConfig must be given at least one key")

    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset:
        filename = self.input_file.replace("<IENS>", str(iens))
        _, keys, time_map, data = read_summary(f"{run_path}/{filename}", self.keys)
        if len(data) == 0 or len(keys) == 0:
            # https://github.com/equinor/ert/issues/6974
            # There is a bug with storing empty responses so we have
            # to raise an error in that case
            raise ValueError(
                f"Did not find any summary values matching {self.keys} in {filename}"
            )
        ds = xr.Dataset(
            {"values": (["name", "time"], data)},
            coords={"time": time_map, "name": keys},
        )
        return ds.drop_duplicates("time")
