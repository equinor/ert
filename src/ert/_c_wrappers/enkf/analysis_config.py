import logging
from math import ceil
from os.path import realpath
from typing import Dict, List, Optional

from ert._c_wrappers.analysis import AnalysisMode, AnalysisModule
from ert._c_wrappers.enkf.analysis_iter_config import AnalysisIterConfig
from ert._c_wrappers.enkf.config_keys import ConfigKeys

logger = logging.getLogger(__name__)


class AnalysisConfigError(Exception):
    pass


class AnalysisConfig:
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
        analysis_iter_config=None,
        analysis_copy=None,
        analysis_set_var=None,
        analysis_select=None,
    ):
        self._rerun = rerun
        self._rerun_start = rerun_start
        self._max_runtime = max_runtime
        self._min_realization = min_realization
        self._global_std_scaling = global_std_scaling
        self._stop_long_running = stop_long_running
        self._alpha = alpha
        self._std_cutoff = std_cutoff
        self._analysis_iter_config = analysis_iter_config or AnalysisIterConfig()
        self._update_log_path = update_log_path

        self._analysis_set_var = analysis_set_var or []
        self._analysis_copy = analysis_copy or []
        es_module = AnalysisModule.ens_smoother_module()
        ies_module = AnalysisModule.iterated_ens_smoother_module()
        self._modules: Dict[str, AnalysisModule] = {
            AnalysisMode.ENSEMBLE_SMOOTHER: es_module,
            AnalysisMode.ITERATED_ENSEMBLE_SMOOTHER: ies_module,
        }
        self._active_module = analysis_select or AnalysisMode.ENSEMBLE_SMOOTHER
        self._copy_modules()
        self._set_modules_var_list()

    def _copy_modules(self):
        for element in self._analysis_copy:
            if isinstance(element, list):
                src_name, dst_name = element
            else:
                src_name = element[ConfigKeys.SRC_NAME]
                dst_name = element[ConfigKeys.DST_NAME]

            module = self._modules.get(src_name)
            if module is not None:
                if module.mode == AnalysisMode.ENSEMBLE_SMOOTHER:
                    new_module = AnalysisModule.ens_smoother_module(dst_name)
                else:
                    new_module = AnalysisModule.iterated_ens_smoother_module(dst_name)
                self._modules[dst_name] = new_module
            else:
                raise AnalysisConfigError(
                    f"Trying to copy module {src_name}" f" which does not exist"
                )

    def _set_modules_var_list(self):
        for set_var in self._analysis_set_var:
            if isinstance(set_var, list):
                module_name, var_name, value = set_var
            else:
                module_name = set_var[ConfigKeys.MODULE_NAME]
                var_name = set_var[ConfigKeys.VAR_NAME]
                value = set_var[ConfigKeys.VALUE]

            module = self.get_module(module_name)
            module.set_var(var_name, value)

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

    def get_stop_long_running(self) -> bool:
        return self._stop_long_running

    def set_stop_long_running(self, stop_long_running: bool):
        self._stop_long_running = stop_long_running

    def get_max_runtime(self) -> int:
        return self._max_runtime

    def set_max_runtime(self, max_runtime: int):
        self._max_runtime = max_runtime

    def active_module_name(self) -> str:
        return self._active_module

    def get_module_list(self) -> List[str]:
        return list(self._modules.keys())

    def get_module(self, module_name: str) -> AnalysisModule:
        if module_name in self._modules:
            return self._modules[module_name]
        raise AnalysisConfigError(f"Analysis module {module_name} not found!")

    def select_module(self, module_name: str) -> bool:
        if module_name in self._modules:
            self._active_module = module_name
            return True
        logger.warning(
            f"Module {module_name} not found."
            f" Active module {self._active_module} not changed"
        )
        return False

    def get_active_module(self) -> AnalysisModule:
        return self._modules[self._active_module]

    def set_global_std_scaling(self, std_scaling: float):
        self._global_std_scaling = std_scaling

    def get_global_std_scaling(self) -> float:
        return self._global_std_scaling

    @property
    def minimum_required_realizations(self) -> int:
        return self._min_realization

    def have_enough_realisations(self, realizations) -> bool:
        return realizations >= self.minimum_required_realizations

    @property
    def case_format(self) -> Optional[str]:
        return self._analysis_iter_config.iter_case

    def case_format_is_set(self) -> bool:
        return self._analysis_iter_config.iter_case is not None

    def set_case_format(self, case_fmt: str):
        self._analysis_iter_config.iter_case = case_fmt

    @property
    def num_retries_per_iter(self) -> int:
        return self._analysis_iter_config.iter_retry_count

    @property
    def num_iterations(self) -> int:
        return self._analysis_iter_config.iter_count

    def set_num_iterations(self, num_iterations: int):
        self._analysis_iter_config.iter_count = num_iterations

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return (
            "AnalysisConfig("
            f"alpha={self._alpha}"
            f"rerun={self._rerun}"
            f"std_cutoff={self._std_cutoff}"
            f"stop_long_running={self._stop_long_running}"
            f"global_std_scaling={self._global_std_scaling}"
            f"max_runtime={self._max_runtime}"
            f"min_realization={self._min_realization}"
            f"update_log_path={self._update_log_path}"
            f"analysis_iter_config={self._analysis_iter_config}"
            f"analysis_copy={self._analysis_copy}"
            f"analysis_set_var={self._analysis_set_var}"
            f"analysis_select={self._active_module})"
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

        if set(self.get_module_list()) != set(other.get_module_list()):
            return False

        if self._active_module != other._active_module:
            return False

        if self._analysis_iter_config != other._analysis_iter_config:
            return False

        if self.minimum_required_realizations != other.minimum_required_realizations:
            return False

        # compare each module
        for a in self.get_module_list():
            if self.get_module(a) != other.get_module(a):
                return False

        return True
