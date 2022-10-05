from math import ceil
from os.path import realpath
from typing import List

from cwrap import BaseCClass

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.analysis import AnalysisModule
from ert._c_wrappers.enkf.analysis_iter_config import AnalysisIterConfig
from ert._c_wrappers.enkf.config_keys import ConfigKeys


class AnalysisConfig(BaseCClass):
    _alloc = ResPrototype("void* analysis_config_alloc()", bind=False)

    _add_module_copy = ResPrototype(
        "void analysis_config_add_module_copy( analysis_config, char* , char* )"
    )

    _free = ResPrototype("void analysis_config_free( analysis_config )")
    _get_active_module_name = ResPrototype(
        "char* analysis_config_get_active_module_name(analysis_config)"
    )
    _get_module = ResPrototype(
        "analysis_module_ref analysis_config_get_module(analysis_config, char*)"
    )
    _select_module = ResPrototype(
        "bool analysis_config_select_module(analysis_config, char*)"
    )
    _has_module = ResPrototype(
        "bool analysis_config_has_module(analysis_config, char*)"
    )

    def __init__(
        self,
        alpha=3.0,
        rerun=False,
        rerun_start=0,
        std_cutoff=1e-6,
        stop_long_running=False,
        global_std_scaling=1.0,
        max_runtime=0,
        min_realization=0,
        update_log_path=None,
        analysis_iter_config=AnalysisIterConfig(),
        analysis_copy=None,
        analysis_set_var=None,
        analysis_select=None,
    ):

        c_ptr = self._alloc()
        self._rerun = rerun
        self._rerun_start = rerun_start
        self._max_runtime = max_runtime
        self._min_realization = min_realization
        self._global_std_scaling = global_std_scaling
        self._stop_long_running = stop_long_running
        self._alpha = alpha
        self._std_cutoff = std_cutoff
        self._analysis_iter_config = analysis_iter_config
        self._update_log_path = update_log_path

        self._analysis_select = analysis_select
        self._analysis_set_var = analysis_set_var or []
        self._analysis_copy = analysis_copy or []

        if c_ptr:
            super().__init__(c_ptr)
            # copy modules
            for element in self._analysis_copy:
                if isinstance(element, list):
                    src_name, dst_name = element
                else:
                    src_name = element[ConfigKeys.SRC_NAME]
                    dst_name = element[ConfigKeys.DST_NAME]

                self._add_module_copy(
                    src_name,
                    dst_name,
                )

            # set var list
            for set_var in self._analysis_set_var:
                if isinstance(set_var, list):
                    module_name, var_name, value = set_var
                else:
                    module_name = set_var[ConfigKeys.MODULE_NAME]
                    var_name = set_var[ConfigKeys.VAR_NAME]
                    value = set_var[ConfigKeys.VALUE]

                module = self._get_module(module_name)
                module._set_var(var_name, str(value))

            if self._analysis_select:
                self._select_module(self._analysis_select)

        else:
            raise ValueError("Failed to construct AnalysisConfig")

    @classmethod
    def from_dict(cls, config_dict) -> "AnalysisConfig":
        num_realization = config_dict.get(ConfigKeys.NUM_REALIZATIONS)
        min_realization = config_dict.get(ConfigKeys.MIN_REALIZATIONS, 0)
        if isinstance(min_realization, str):
            if "%" in min_realization:
                min_realization = ceil(
                    num_realization * float(min_realization.strip("%")) / 100
                )
            else:
                min_realization = int(min_realization)
        # Make sure min_realization is not greater than num_realization
        if min_realization == 0:
            min_realization = num_realization
        min_realization = min(min_realization, num_realization)

        config = cls(
            alpha=config_dict.get(ConfigKeys.ALPHA_KEY, 3.0),
            rerun=config_dict.get(ConfigKeys.RERUN_KEY, False),
            rerun_start=config_dict.get(ConfigKeys.RERUN_START_KEY, 0),
            std_cutoff=config_dict.get(ConfigKeys.STD_CUTOFF_KEY, 1e-6),
            stop_long_running=config_dict.get(ConfigKeys.STOP_LONG_RUNNING, False),
            global_std_scaling=config_dict.get(ConfigKeys.GLOBAL_STD_SCALING, 1.0),
            max_runtime=config_dict.get(ConfigKeys.MAX_RUNTIME, 0),
            min_realization=min_realization,
            update_log_path=config_dict.get(ConfigKeys.UPDATE_LOG_PATH, "update_log"),
            analysis_iter_config=AnalysisIterConfig.from_dict(config_dict),
            analysis_copy=config_dict.get(ConfigKeys.ANALYSIS_COPY, []),
            analysis_set_var=config_dict.get(ConfigKeys.ANALYSIS_SET_VAR, []),
            analysis_select=config_dict.get(ConfigKeys.ANALYSIS_SELECT),
        )
        return config

    def get_rerun(self):
        return self._rerun

    def set_rerun(self, rerun):
        self._rerun = rerun

    def get_rerun_start(self):
        return self._rerun_start

    def set_rerun_start(self, index):
        self._rerun_start = index

    def get_log_path(self) -> str:
        return realpath(self._update_log_path)

    def set_log_path(self, path: str):
        self._update_log_path = path

    def get_enkf_alpha(self) -> float:
        return self._alpha

    def set_enkf_alpha(self, alpha: float):
        self._alpha = alpha

    def get_std_cutoff(self) -> float:
        return self._std_cutoff

    def set_std_cutoff(self, std_cutoff: float):
        self._std_cutoff = std_cutoff

    def get_analysis_iter_config(self) -> AnalysisIterConfig:
        return self._analysis_iter_config

    def get_stop_long_running(self) -> bool:
        return self._stop_long_running

    def set_stop_long_running(self, stop_long_running: bool):
        self._stop_long_running = stop_long_running

    def get_max_runtime(self) -> int:
        return self._max_runtime

    def set_max_runtime(self, max_runtime: int):
        self._max_runtime = max_runtime

    def free(self):
        self._free()

    def activeModuleName(self) -> str:
        return self._get_active_module_name()

    def getModuleList(self) -> List[str]:
        return _clib.analysis_config_module_names(self)

    def getModule(self, module_name: str) -> AnalysisModule:
        return self._get_module(module_name)

    def hasModule(self, module_name: str) -> bool:
        return self._has_module(module_name)

    def selectModule(self, module_name: str) -> bool:
        return self._select_module(module_name)

    def getActiveModule(self) -> AnalysisModule:
        return self.getModule(self.activeModuleName())

    def set_global_std_scaling(self, std_scaling: float):
        self._global_std_scaling = std_scaling

    def get_global_std_scaling(self) -> float:
        return self._global_std_scaling

    @property
    def minimum_required_realizations(self) -> int:
        return self._min_realization

    def have_enough_realisations(self, realizations) -> bool:
        return realizations >= self.minimum_required_realizations

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        if not self._address():
            return "<AnalysisConfig()>"
        return (
            "AnalysisConfig(config_dict={"
            f"'UPDATE_LOG_PATH': {self.get_log_path()}, "
            f"'MAX_RUNTIME': {self.get_max_runtime()}, "
            f"'GLOBAL_STD_SCALING': {self.get_global_std_scaling()}, "
            f"'STOP_LONG_RUNNING': {self.get_stop_long_running()}, "
            f"'STD_CUTOFF': {self.get_std_cutoff()}, "
            f"'ENKF_ALPHA': {self.get_enkf_alpha()}, "
            f"'RERUN': {self.get_rerun()}, "
            f"'RERUN_START': {self.get_rerun_start()}, "
            f"'ANALYSIS_SELECT': {self.activeModuleName()}, "
            f"'MODULE_LIST': {self.getModuleList()}, "
            f"'ITER_CONFIG': {self.get_analysis_iter_config()}, "
            f"'MIN_REALIZATIONS_REQUIRED': {self.minimum_required_realizations}, "
            "})"
        )

    def __eq__(self, other):
        if realpath(self.get_log_path()) != realpath(other.get_log_path()):
            return False

        if self.get_max_runtime() != other.get_max_runtime():
            return False

        if self.get_global_std_scaling() != other.get_global_std_scaling():
            return False

        if self.get_stop_long_running() != other.get_stop_long_running():
            return False

        if self.get_std_cutoff() != other.get_std_cutoff():
            return False

        if self.get_enkf_alpha() != other.get_enkf_alpha():
            return False

        if self.get_rerun() != other.get_rerun():
            return False

        if self.get_rerun_start() != other.get_rerun_start():
            return False

        if set(self.getModuleList()) != set(other.getModuleList()):
            return False

        if self.activeModuleName() != other.activeModuleName():
            return False

        if self.get_analysis_iter_config() != other.get_analysis_iter_config():
            return False

        if self.minimum_required_realizations != other.minimum_required_realizations:
            return False

        # compare each module
        for a in self.getModuleList():
            if self.getModule(a) != other.getModule(a):
                return False

        return True
