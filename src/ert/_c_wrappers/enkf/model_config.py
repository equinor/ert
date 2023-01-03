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
        num_realizations: int,
        refcase: Optional[EclSum] = None,
        data_root: Optional[str] = None,
        history_source: Optional[HistorySourceEnum] = None,
        runpath_format_string: Optional[str] = None,
        jobname_format_string: Optional[str] = None,
        gen_kw_export_name: Optional[str] = None,
        obs_config_file: Optional[str] = None,
        time_map_file: Optional[str] = None,
    ):
        self.num_realizations = num_realizations
        self.refcase = refcase
        self.data_root = data_root

        self.history_source = (
            history_source
            if self.refcase is not None and history_source is not None
            else self.DEFAULT_HISTORY_SOURCE
        )

        if runpath_format_string is None:
            self.runpath_format_string = self.DEFAULT_RUNPATH
        elif "%d" in runpath_format_string:
            self.runpath_format_string = runpath_format_string
            logger.warning(
                "RUNPATH keyword should use syntax "
                f"`{self.DEFAULT_RUNPATH}` "
                "instead of deprecated syntax "
                f"`{runpath_format_string}`"
            )
        elif not any(x in runpath_format_string for x in ["<ITER>, <IENS>"]):
            self.runpath_format_string = runpath_format_string
            logger.error(
                "RUNPATH keyword should use syntax " f"`{self.DEFAULT_RUNPATH}`."
            )

        self.jobname_format_string = jobname_format_string
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
            num_realizations=config_dict.get(ConfigKeys.NUM_REALIZATIONS),
            refcase=refcase,
            data_root=config_dict.get(ConfigKeys.DATAROOT),
            history_source=HistorySourceEnum.from_string(
                config_dict.get(ConfigKeys.HISTORY_SOURCE)
            )
            if ConfigKeys.HISTORY_SOURCE in config_dict
            else None,
            runpath_format_string=config_dict.get(ConfigKeys.RUNPATH),
            jobname_format_string=config_dict.get(
                ConfigKeys.JOBNAME, config_dict.get(ConfigKeys.ECLBASE, None)
            ),
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
            f"data_root: {self.data_root},\n"
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
                self.data_root == other.data_root,
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
        return not self == other
