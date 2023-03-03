import logging
from typing import Optional

from ecl.summary import EclSum

from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.enkf.time_map import TimeMap
from ert._c_wrappers.sched import HistorySourceEnum

logger = logging.getLogger(__name__)


class ModelConfig:
    DEFAULT_HISTORY_SOURCE = HistorySourceEnum.REFCASE_HISTORY
    DEFAULT_RUNPATH = "simulations/realization-<IENS>/iter-<ITER>"
    DEFAULT_GEN_KW_EXPORT_NAME = "parameters"

    def __init__(
        self,
        num_realizations: int = 1,
        refcase: Optional[EclSum] = None,
        history_source: Optional[HistorySourceEnum] = None,
        runpath_format_string: Optional[str] = None,
        jobname_format_string: Optional[str] = None,
        runpath_file: str = ".ert_runpath_list",
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
            replace_runpath_format(jobname_format_string) or "JOB<IENS>"
        )
        self.runpath_format_string = (
            replace_runpath_format(runpath_format_string) or self.DEFAULT_RUNPATH
        )
        self.runpath_file = runpath_file

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
            self.time_map = TimeMap()
            try:
                self.time_map.read_text(time_map_file)
            except ValueError as err:
                logger.warning(err)
            except IOError as err:
                logger.warning(f"failed to load timemap - {err}")

    @classmethod
    def from_dict(cls, refcase: EclSum, config_dict: dict) -> "ModelConfig":
        return cls(
            num_realizations=config_dict.get(ConfigKeys.NUM_REALIZATIONS, 1),
            refcase=refcase,
            history_source=HistorySourceEnum.from_string(
                config_dict.get(ConfigKeys.HISTORY_SOURCE)
            )
            if ConfigKeys.HISTORY_SOURCE in config_dict
            else None,
            runpath_format_string=config_dict.get(ConfigKeys.RUNPATH),
            jobname_format_string=config_dict.get(
                ConfigKeys.JOBNAME, config_dict.get(ConfigKeys.ECLBASE, None)
            ),
            runpath_file=config_dict.get(ConfigKeys.RUNPATH_FILE, ".ert_runpath_list"),
            gen_kw_export_name=config_dict.get(ConfigKeys.GEN_KW_EXPORT_NAME),
            obs_config_file=config_dict.get(ConfigKeys.OBS_CONFIG),
            time_map_file=config_dict.get(ConfigKeys.TIME_MAP),
        )

    def get_history_num_steps(self) -> int:
        if self.refcase:
            return self.refcase.last_report + 1
        if self.time_map:
            return len(self.time_map)
        return 0

    def __repr__(self):
        return f"ModelConfig(\n{self}\n)"

    def __str__(self):
        return (
            f"num_realizations: {self.num_realizations},\n"
            f"refcase: {self.refcase},\n"
            f"history_source: {self.history_source},\n"
            f"runpath_format_string: {self.runpath_format_string},\n"
            f"jobname_format_string: {self.jobname_format_string},\n"
            f"gen_kw_export_name: {self.gen_kw_export_name},\n"
            f"obs_config_file: {self.obs_config_file},\n"
            f"time_map_file: {self._time_map_file}"
        )

    def __eq__(self, other):
        return all(
            [
                self.num_realizations == other.num_realizations,
                self.history_source == other.history_source,
                self.runpath_format_string == other.runpath_format_string,
                self.jobname_format_string == other.jobname_format_string,
                self.gen_kw_export_name == other.gen_kw_export_name,
                self.obs_config_file == other.obs_config_file,
                self._time_map_file == other._time_map_file,
                self.time_map == other.time_map,
            ]
        )

    def __ne__(self, other):
        return self != other


def replace_runpath_format(format_string: Optional[str]) -> Optional[str]:
    if format_string is None:
        return format_string
    format_string = format_string.replace("%d", "<IENS>", 1)
    format_string = format_string.replace("%d", "<ITER>", 1)
    return format_string
