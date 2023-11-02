from __future__ import annotations

import logging
import os.path
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from ._config_values import (
    DEFAULT_GEN_KW_EXPORT_NAME,
    DEFAULT_HISTORY_SOURCE,
    DEFAULT_RUNPATH,
    ErtConfigValues,
)
from .parsing import ConfigValidationError, HistorySource

if TYPE_CHECKING:
    from typing import List

logger = logging.getLogger(__name__)


DEFAULT_JOBNAME_FORMAT = "<CONFIG_FILE>-<IENS>"
DEFAULT_ECLBASE_FORMAT = "ECLBASE<IENS>"


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


class ModelConfig:
    def __init__(
        self,
        num_realizations: int = 1,
        history_source: HistorySource = DEFAULT_HISTORY_SOURCE,
        runpath_format_string: str = DEFAULT_RUNPATH,
        jobname_format_string: Optional[str] = None,
        eclbase_format_string: Optional[str] = None,
        gen_kw_export_name: str = DEFAULT_GEN_KW_EXPORT_NAME,
        obs_config_file: Optional[str] = None,
        time_map_file: Optional[str] = None,
    ):
        self.num_realizations = num_realizations
        self.history_source = history_source
        if jobname_format_string is None:
            self.jobname_format_string = (
                eclbase_format_string
                if eclbase_format_string is not None
                else DEFAULT_JOBNAME_FORMAT
            )
        else:
            self.jobname_format_string = jobname_format_string
        if eclbase_format_string is None:
            self.eclbase_format_string = (
                jobname_format_string
                if jobname_format_string is not None
                else DEFAULT_ECLBASE_FORMAT
            )
        else:
            self.eclbase_format_string = eclbase_format_string
        self.jobname_format_string = _replace_runpath_format(self.jobname_format_string)
        self.eclbase_format_string = _replace_runpath_format(self.eclbase_format_string)
        self.runpath_format_string = _replace_runpath_format(runpath_format_string)

        if self.runpath_format_string is not None and not any(
            x in self.runpath_format_string for x in ["<ITER>", "<IENS>"]
        ):
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

    @classmethod
    def from_values(cls, config_values: ErtConfigValues) -> "ModelConfig":
        return cls(
            num_realizations=config_values.num_realizations,
            history_source=config_values.history_source,
            runpath_format_string=config_values.runpath,
            jobname_format_string=config_values.jobname,
            eclbase_format_string=config_values.eclbase,
            gen_kw_export_name=config_values.gen_kw_export_name,
            obs_config_file=config_values.obs_config,
            time_map_file=config_values.time_map,
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
