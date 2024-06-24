from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np

from .. import ConfigWarning, ResponseTypes, SummaryObservation
from ..parsing import ContextList, ContextValue, HistorySource
from ..parsing.observations_parser import (
    HistoryValues,
    ObservationConfigError,
    SummaryValues,
)
from . import obs_commons
from .general_observation import GenObservation
from .observation_vector import ObsVector
from .response_config import ObsArgs, ResponseConfigWithLifecycleHooks


def history_key(key: str) -> str:
    keyword, *rest = key.split(":")
    return ":".join([keyword + "H"] + rest)


class SummaryConfigWithHooks(ResponseConfigWithLifecycleHooks):
    @classmethod
    def from_config_list(
        cls, config_list: List[ContextList[ContextValue]]
    ) -> Union[
        "ResponseConfigWithLifecycleHooks", List["ResponseConfigWithLifecycleHooks"]
    ]:
        pass

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
                ResponseTypes.summary,
                obs_name,
                obs_name,
                data,
            )
        }

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
                ResponseTypes.summary,
                obs_key,
                summary_key,
                {date: SummaryObservation(summary_key, obs_key, value, std_dev)},
            )
        }

    @staticmethod
    def parse_observation_from_legacy_obsconfig(args: ObsArgs) -> Dict[str, ObsVector]:
        if type(args.values) is HistoryValues:
            return SummaryConfigWithHooks._parse_history_obs(args)

        elif type(args.values) == SummaryValues:
            return SummaryConfigWithHooks._parse_summary_obs(args)

        raise KeyError(
            f"Expected history or summary observation, got {type(args.values)}"
        )

    @classmethod
    def response_type(cls) -> str:
        return "SUMMARY"

    @classmethod
    def ert_config_response_keyword(cls) -> str:
        return "SUMMARY"

    @classmethod
    def ert_config_observation_keyword(cls) -> List[str]:
        """
        These refer to SUMMARY_OBSERVATION / HISTORY_OBSERVATION
        declared directly in the ert config, and not
        that declared within the OBS_CONFIG
        """
        return ["SUMMARY_OBSERVATION", "HISTORY_OBSERVATION"]

    def parse_response_from_config(self, config_list: List[Tuple[str, str]]) -> None:
        pass

    def parse_observation_from_config(
        self, config_list: List[Tuple[str, str]], obs_args: ObsArgs
    ) -> None:
        pass

    def parse_response_from_runpath(self, run_path: str) -> str:
        pass
