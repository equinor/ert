from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional, no_type_check

from ecl.summary import EclSum

from .history_source import HistorySource
from .parsing import ConfigDict, ConfigKeys

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


class ModelConfig:  # pylint: disable=too-many-instance-attributes
    DEFAULT_HISTORY_SOURCE = HistorySource.REFCASE_HISTORY
    DEFAULT_RUNPATH = "simulations/realization-<IENS>/iter-<ITER>"
    DEFAULT_GEN_KW_EXPORT_NAME = "parameters"

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_realizations: int = 1,
        refcase: Optional[EclSum] = None,
        history_source: Optional[HistorySource] = None,
        runpath_format_string: Optional[str] = None,
        jobname_format_string: Optional[str] = None,
        eclbase_format_string: Optional[str] = None,
        gen_kw_export_name: Optional[str] = None,
        obs_config_file: Optional[str] = None,
        time_map_file: Optional[str] = None,
    ):
        self.num_realizations = num_realizations
        self.refcase = refcase
        self.history_source = (
            history_source
            if self.refcase is not None and history_source is not None
            else self.DEFAULT_HISTORY_SOURCE
        )
        self.jobname_format_string = (
            replace_runpath_format(jobname_format_string) or "<CONFIG_FILE>-<IENS>"
        )
        self.eclbase_format_string = (
            replace_runpath_format(eclbase_format_string) or "ECLBASE<IENS>"
        )
        self.runpath_format_string = (
            replace_runpath_format(runpath_format_string) or self.DEFAULT_RUNPATH
        )

        if self.runpath_format_string is not None and not any(
            x in self.runpath_format_string for x in ["<ITER>", "<IENS>"]
        ):
            logger.warning(
                "RUNPATH keyword contains no value placeholders: "
                f"`{runpath_format_string}`. Valid example: "
                f"`{self.DEFAULT_RUNPATH}` "
            )

        self.gen_kw_export_name = (
            gen_kw_export_name
            if gen_kw_export_name is not None
            else self.DEFAULT_GEN_KW_EXPORT_NAME
        )
        self.obs_config_file = obs_config_file
        self.time_map = None
        self._time_map_file = time_map_file

        if time_map_file is not None:
            try:
                self.time_map = _read_time_map(time_map_file)
            except ValueError as err:
                logger.warning(err)
            except IOError as err:
                logger.warning(f"failed to load timemap - {err}")

    @no_type_check
    @classmethod
    def from_dict(
        cls, refcase: Optional[EclSum], config_dict: ConfigDict
    ) -> "ModelConfig":
        return cls(
            num_realizations=config_dict.get(ConfigKeys.NUM_REALIZATIONS, 1),
            refcase=refcase,
            history_source=HistorySource[
                str(config_dict.get(ConfigKeys.HISTORY_SOURCE))
            ]
            if ConfigKeys.HISTORY_SOURCE in config_dict
            else None,
            runpath_format_string=config_dict.get(ConfigKeys.RUNPATH),
            jobname_format_string=config_dict.get(
                ConfigKeys.JOBNAME, config_dict.get(ConfigKeys.ECLBASE)
            ),
            eclbase_format_string=config_dict.get(
                ConfigKeys.ECLBASE, config_dict.get(ConfigKeys.JOBNAME)
            ),
            gen_kw_export_name=config_dict.get(ConfigKeys.GEN_KW_EXPORT_NAME),
            obs_config_file=config_dict.get(ConfigKeys.OBS_CONFIG),
            time_map_file=config_dict.get(ConfigKeys.TIME_MAP),
        )

    def __repr__(self) -> str:
        return f"ModelConfig(\n{self}\n)"

    def __str__(self) -> str:
        return (
            f"num_realizations: {self.num_realizations},\n"
            f"refcase: {self.refcase},\n"
            f"history_source: {self.history_source},\n"
            f"runpath_format_string: {self.runpath_format_string},\n"
            f"jobname_format_string: {self.jobname_format_string},\n"
            f"eclbase_format_string: {self.eclbase_format_string},\n"
            f"gen_kw_export_name: {self.gen_kw_export_name},\n"
            f"obs_config_file: {self.obs_config_file},\n"
            f"time_map_file: {self._time_map_file}"
        )

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


def replace_runpath_format(format_string: Optional[str]) -> Optional[str]:
    if format_string is None:
        return format_string
    format_string = format_string.replace("%d", "<IENS>", 1)
    format_string = format_string.replace("%d", "<ITER>", 1)
    return format_string
