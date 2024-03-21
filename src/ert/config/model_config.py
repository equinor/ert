from __future__ import annotations

import logging
import os.path
from datetime import datetime
from typing import TYPE_CHECKING, Optional, no_type_check

from .parsing import ConfigDict, ConfigKeys, ConfigValidationError, HistorySource

if TYPE_CHECKING:
    from typing import List

logger = logging.getLogger(__name__)


def _read_time_map(file_name: str) -> List[datetime]:
    def str_to_datetime(date_str: str) -> datetime:
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            logger.warning(
                "DD/MM/YYYY date format is deprecated"
                ", please use ISO date format YYYY-MM-DD."
            )
            return datetime.strptime(date_str, "%d/%m/%Y")

    dates = []
    with open(file_name, "r", encoding="utf-8") as fin:
        for line in fin:
            dates.append(str_to_datetime(line.strip()))
    return dates


DEFAULT_HISTORY_SOURCE = HistorySource.REFCASE_HISTORY
DEFAULT_RUNPATH = "simulations/realization-<IENS>/iter-<ITER>"
DEFAULT_GEN_KW_EXPORT_NAME = "parameters"
DEFAULT_JOBNAME_FORMAT = "<CONFIG_FILE>-<IENS>"
DEFAULT_ECLBASE_FORMAT = "ECLBASE<IENS>"


class ModelConfig:
    def __init__(
        self,
        num_realizations: int = 1,
        history_source: HistorySource = DEFAULT_HISTORY_SOURCE,
        runpath_format_string: str = DEFAULT_RUNPATH,
        jobname_format_string: str = DEFAULT_JOBNAME_FORMAT,
        eclbase_format_string: str = DEFAULT_ECLBASE_FORMAT,
        gen_kw_export_name: str = DEFAULT_GEN_KW_EXPORT_NAME,
        obs_config_file: Optional[str] = None,
        time_map_file: Optional[str] = None,
    ):
        self.num_realizations = num_realizations
        self.history_source = history_source
        self.jobname_format_string = _replace_runpath_format(jobname_format_string)
        self.eclbase_format_string = _replace_runpath_format(eclbase_format_string)

        # do not combine styles
        if "%d" in runpath_format_string and any(
            x in runpath_format_string for x in ["<ITER>", "<IENS>"]
        ):
            raise ConfigValidationError(
                f"RUNPATH cannot combine deprecated and new style placeholders: `{runpath_format_string}`. Valid example `{DEFAULT_RUNPATH}`"
            )

        # do not allow multiple occurrences
        for kw in ["<ITER>", "<IENS>"]:
            if runpath_format_string.count(kw) > 1:
                raise ConfigValidationError(
                    f"RUNPATH cannot contain multiple {kw} placeholders: `{runpath_format_string}`. Valid example `{DEFAULT_RUNPATH}`"
                )

        # do not allow too many placeholders
        if runpath_format_string.count("%d") > 2:
            raise ConfigValidationError(
                f"RUNPATH cannot contain more than two value placeholders: `{runpath_format_string}`. Valid example `{DEFAULT_RUNPATH}`"
            )

        if "/" in self.jobname_format_string:
            raise ConfigValidationError.with_context(
                "JOBNAME cannot contain '/'.", jobname_format_string
            )

        self.runpath_format_string = _replace_runpath_format(runpath_format_string)

        if not any(x in self.runpath_format_string for x in ["<ITER>", "<IENS>"]):
            logger.warning(
                "RUNPATH keyword contains no value placeholders: "
                f"`{runpath_format_string}`. Valid example: "
                f"`{DEFAULT_RUNPATH}` "
            )

        self.gen_kw_export_name = gen_kw_export_name
        self.obs_config_file = obs_config_file
        self.time_map = None
        self._time_map_file = (
            os.path.abspath(time_map_file) if time_map_file is not None else None
        )

        if time_map_file is not None:
            try:
                self.time_map = _read_time_map(time_map_file)
            except (ValueError, IOError) as err:
                raise ConfigValidationError.with_context(
                    f"Could not read timemap file {time_map_file}: {err}", time_map_file
                ) from err

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> "ModelConfig":
        return cls(
            num_realizations=config_dict.get(ConfigKeys.NUM_REALIZATIONS, 1),
            history_source=config_dict.get(
                ConfigKeys.HISTORY_SOURCE, DEFAULT_HISTORY_SOURCE
            ),
            runpath_format_string=config_dict.get(ConfigKeys.RUNPATH, DEFAULT_RUNPATH),
            jobname_format_string=config_dict.get(
                ConfigKeys.JOBNAME,
                os.path.basename(
                    config_dict.get(ConfigKeys.ECLBASE, DEFAULT_JOBNAME_FORMAT)
                ),
            ),
            eclbase_format_string=config_dict.get(
                ConfigKeys.ECLBASE,
                config_dict.get(ConfigKeys.JOBNAME, DEFAULT_ECLBASE_FORMAT),
            ),
            gen_kw_export_name=config_dict.get(
                ConfigKeys.GEN_KW_EXPORT_NAME, DEFAULT_GEN_KW_EXPORT_NAME
            ),
            obs_config_file=config_dict.get(ConfigKeys.OBS_CONFIG),
            time_map_file=config_dict.get(ConfigKeys.TIME_MAP),
        )

    def __repr__(self) -> str:
        return (
            "ModelConfig("
            f"num_realizations={self.num_realizations}, "
            f"history_source={self.history_source}, "
            f"runpath_format_string={self.runpath_format_string}, "
            f"jobname_format_string={self.jobname_format_string}, "
            f"eclbase_format_string={self.eclbase_format_string}, "
            f"gen_kw_export_name={self.gen_kw_export_name}, "
            f"obs_config_file={self.obs_config_file}, "
            f"time_map_file={self._time_map_file}"
            ")"
        )

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelConfig):
            return False
        return all(
            [
                self.num_realizations == other.num_realizations,
                self.history_source == other.history_source,
                self.runpath_format_string == other.runpath_format_string,
                self.jobname_format_string == other.jobname_format_string,
                self.eclbase_format_string == other.eclbase_format_string,
                self.gen_kw_export_name == other.gen_kw_export_name,
                self.obs_config_file == other.obs_config_file,
                self._time_map_file == other._time_map_file,
                self.time_map == other.time_map,
            ]
        )


def _replace_runpath_format(format_string: str) -> str:
    format_string = format_string.replace("%d", "<IENS>", 1)
    format_string = format_string.replace("%d", "<ITER>", 1)
    return format_string
