import logging
import warnings
from math import ceil
from os.path import realpath
from typing import Dict, List, Optional, Tuple, no_type_check

from .analysis_iter_config import AnalysisIterConfig
from .analysis_module import AnalysisMode, AnalysisModule
from .parsing import ConfigDict, ConfigKeys, ConfigValidationError, ConfigWarning

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class AnalysisConfig:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        alpha: float = 3.0,
        std_cutoff: float = 1e-6,
        stop_long_running: bool = False,
        max_runtime: int = 0,
        min_realization: int = 0,
        update_log_path: str = "update_log",
        analysis_iter_config: Optional[AnalysisIterConfig] = None,
        analysis_set_var: Optional[List[Tuple[str, str, str]]] = None,
        analysis_select: Optional[str] = None,
    ) -> None:
        self._max_runtime = max_runtime
        self.minimum_required_realizations = min_realization
        self._stop_long_running = stop_long_running
        self._alpha = alpha
        self._std_cutoff = std_cutoff
        self._analysis_iter_config = analysis_iter_config or AnalysisIterConfig()
        self._update_log_path = update_log_path
        self._min_realization = min_realization

        self._analysis_set_var = analysis_set_var or []
        es_module = AnalysisModule.ens_smoother_module()
        ies_module = AnalysisModule.iterated_ens_smoother_module()
        self._modules: Dict[str, AnalysisModule] = {
            AnalysisMode.ENSEMBLE_SMOOTHER: es_module,
            AnalysisMode.ITERATED_ENSEMBLE_SMOOTHER: ies_module,
        }
        self._active_module = analysis_select or AnalysisMode.ENSEMBLE_SMOOTHER
        self._set_modules_var_list()

    def _set_modules_var_list(self) -> None:
        for module_name, var_name, value in self._analysis_set_var:
            module = self.get_module(module_name)
            module.set_var(var_name, value)

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> "AnalysisConfig":
        num_realization: int = config_dict.get(ConfigKeys.NUM_REALIZATIONS, 1)
        min_realization_str: str = config_dict.get(ConfigKeys.MIN_REALIZATIONS, "0")
        if "%" in min_realization_str:
            try:
                min_realization = ceil(
                    num_realization * float(min_realization_str.strip("%")) / 100
                )
            except ValueError as err:
                raise ConfigValidationError.with_context(
                    f"MIN_REALIZATIONS {min_realization_str!r} contained %"
                    " but was not a valid percentage",
                    min_realization_str,
                ) from err
        else:
            try:
                min_realization = int(min_realization_str)
            except ValueError as err:
                raise ConfigValidationError.with_context(
                    f"MIN_REALIZATIONS value is not integer {min_realization_str!r}",
                    min_realization_str,
                ) from err
        # Make sure min_realization is not greater than num_realization
        if min_realization == 0:
            min_realization = num_realization

        if min_realization > num_realization:
            warnings.warn(
                ConfigWarning.with_context(
                    "MIN_REALIZATIONS set to more than NUM_REALIZATIONS, "
                    "will set required to successful realizations to 100%. "
                    "For more flexibility, you can use e.g. 'MIN_REALIZATIONS 80%'.",
                    min_realization_str,
                ),
                stacklevel=1,
            )
        min_realization = min(min_realization, num_realization)

        config = cls(
            alpha=config_dict.get(ConfigKeys.ENKF_ALPHA, 3.0),
            std_cutoff=config_dict.get(ConfigKeys.STD_CUTOFF, 1e-6),
            stop_long_running=config_dict.get(ConfigKeys.STOP_LONG_RUNNING, False),
            max_runtime=config_dict.get(ConfigKeys.MAX_RUNTIME, 0),
            min_realization=min_realization,
            update_log_path=config_dict.get(ConfigKeys.UPDATE_LOG_PATH, "update_log"),
            analysis_iter_config=AnalysisIterConfig.from_dict(config_dict),
            analysis_set_var=config_dict.get(ConfigKeys.ANALYSIS_SET_VAR, []),
            analysis_select=config_dict.get(ConfigKeys.ANALYSIS_SELECT),
        )
        return config

    @property
    def log_path(self) -> str:
        return realpath(self._update_log_path)

    @log_path.setter
    def log_path(self, log_path: str) -> None:
        self._update_log_path = log_path

    @property
    def enkf_alpha(self) -> float:
        return self._alpha

    @enkf_alpha.setter
    def enkf_alpha(self, value: float) -> None:
        self._alpha = value

    @property
    def std_cutoff(self) -> float:
        return self._std_cutoff

    @property
    def stop_long_running(self) -> bool:
        return self._stop_long_running

    @property
    def max_runtime(self) -> Optional[int]:
        return self._max_runtime if self._max_runtime > 0 else None

    def get_module(self, module_name: str) -> AnalysisModule:
        if module_name in self._modules:
            return self._modules[module_name]
        raise ConfigValidationError(f"Analysis module {module_name} not found!")

    def select_module(self, module_name: str) -> bool:
        if module_name in self._modules:
            self._active_module = module_name
            return True
        logger.warning(
            f"Module {module_name} not found."
            f" Active module {self._active_module} not changed"
        )
        return False

    def active_module(self) -> AnalysisModule:
        return self._modules[self._active_module]

    def have_enough_realisations(self, realizations: int) -> bool:
        return realizations >= self.minimum_required_realizations

    @property
    def case_format(self) -> Optional[str]:
        return self._analysis_iter_config.iter_case

    def case_format_is_set(self) -> bool:
        return self._analysis_iter_config.iter_case is not None

    def set_case_format(self, case_fmt: str) -> None:
        self._analysis_iter_config.iter_case = case_fmt

    @property
    def num_retries_per_iter(self) -> int:
        return self._analysis_iter_config.iter_retry_count

    @property
    def num_iterations(self) -> int:
        return self._analysis_iter_config.iter_count

    def set_num_iterations(self, num_iterations: int) -> None:
        self._analysis_iter_config.iter_count = num_iterations

    def __repr__(self) -> str:
        return (
            "AnalysisConfig("
            f"alpha={self._alpha}, "
            f"std_cutoff={self._std_cutoff}, "
            f"stop_long_running={self._stop_long_running}, "
            f"max_runtime={self._max_runtime}, "
            f"min_realization={self._min_realization}, "
            f"update_log_path={self._update_log_path}, "
            f"analysis_iter_config={self._analysis_iter_config}, "
            f"analysis_set_var={self._analysis_set_var}, "
            f"analysis_select={self._active_module})"
        )

    def __eq__(  # pylint: disable=too-many-return-statements
        self, other: object
    ) -> bool:
        if not isinstance(other, AnalysisConfig):
            return False

        if self.log_path != other.log_path:
            return False

        if self.max_runtime != other.max_runtime:
            return False

        if self.stop_long_running != other.stop_long_running:
            return False

        if self.std_cutoff != other.std_cutoff:
            return False

        if self.enkf_alpha != other.enkf_alpha:
            return False

        if set(self._modules) != set(other._modules):
            return False

        if self._active_module != other._active_module:
            return False

        if self._analysis_iter_config != other._analysis_iter_config:
            return False

        if self.minimum_required_realizations != other.minimum_required_realizations:
            return False

        # compare each module
        return all(self.get_module(a) == other.get_module(a) for a in self._modules)
