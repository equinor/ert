from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import ceil
from os.path import realpath
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union, no_type_check

from pydantic import ValidationError

from .analysis_iter_config import AnalysisIterConfig
from .analysis_module import ESSettings, IESSettings
from .parsing import (
    AnalysisMode,
    ConfigDict,
    ConfigKeys,
    ConfigValidationError,
    ConfigWarning,
)

logger = logging.getLogger(__name__)

DEFAULT_ANALYSIS_MODE = AnalysisMode.ENSEMBLE_SMOOTHER
ObservationGroups = List[str]


@dataclass
class UpdateSettings:
    std_cutoff: float = 1e-6
    alpha: float = 3.0
    auto_scale_observations: List[ObservationGroups] = field(default_factory=list)
    min_required_realizations: int = 2


class AnalysisConfig:
    def __init__(
        self,
        alpha: float = 3.0,
        std_cutoff: float = 1e-6,
        max_runtime: int = 0,
        min_realization: int = 0,
        update_log_path: Union[str, Path] = "update_log",
        analysis_iter_config: Optional[AnalysisIterConfig] = None,
        analysis_set_var: Optional[List[Tuple[str, str, str]]] = None,
    ) -> None:
        self._max_runtime = max_runtime
        self.minimum_required_realizations = min_realization
        self._alpha = alpha
        self._std_cutoff = std_cutoff
        self._analysis_iter_config = analysis_iter_config or AnalysisIterConfig()
        self._update_log_path = Path(update_log_path)
        self._min_realization = min_realization

        options: Dict[str, Dict[str, Any]] = {"STD_ENKF": {}, "IES_ENKF": {}}
        observation_settings: Dict[str, Any] = {
            "alpha": alpha,
            "std_cutoff": std_cutoff,
            "auto_scale_observations": [],
            "min_required_realizations": min_realization,
        }
        analysis_set_var = [] if analysis_set_var is None else analysis_set_var
        inversion_str_map: Final = {
            "STD_ENKF": {
                **dict.fromkeys(["EXACT", "0"], "exact"),
                **dict.fromkeys(["SUBSPACE_EXACT_R", "1"], "subspace"),
                **dict.fromkeys(["SUBSPACE_EE_R", "2"], "subspace"),
                **dict.fromkeys(["SUBSPACE_RE", "3"], "subspace"),
            },
            "IES_ENKF": {
                **dict.fromkeys(["EXACT", "0"], "direct"),
                **dict.fromkeys(["SUBSPACE_EXACT_R", "1"], "subspace_exact"),
                **dict.fromkeys(["SUBSPACE_EE_R", "2"], "subspace_projected"),
                **dict.fromkeys(["SUBSPACE_RE", "3"], "subspace_projected"),
            },
        }
        deprecated_keys = ["ENKF_NCOMP", "ENKF_SUBSPACE_DIMENSION"]
        deprecated_inversion_keys = ["USE_EE", "USE_GE"]
        errors = []
        all_errors = []

        for module_name, var_name, value in analysis_set_var:
            if module_name == "OBSERVATIONS":
                if var_name == "AUTO_SCALE":
                    observation_settings["auto_scale_observations"].append(
                        value.split(",")
                    )
                else:
                    all_errors.append(
                        ConfigValidationError(
                            f"Unknown variable: {var_name} for: ANALYSIS_SET_VAR OBSERVATIONS {var_name}"
                            "Valid options: AUTO_SCALE"
                        )
                    )
                continue
            if var_name in deprecated_keys:
                errors.append(var_name)
                continue
            if var_name == "ENKF_FORCE_NCOMP":
                continue
            if var_name in deprecated_inversion_keys:
                all_errors.append(
                    ConfigValidationError(
                        f"Keyword {var_name} has been replaced by INVERSION and has no effect."
                        "\n\nPlease see https://ert.readthedocs.io/en/latest/reference/configuration/keywords.html#inversion-algorithm "
                        "for documentation how to use this instead."
                    )
                )
                continue
            if var_name in ["INVERSION", "IES_INVERSION"]:
                if value in inversion_str_map[module_name]:
                    new_value = inversion_str_map[module_name][value]
                    if var_name == "IES_INVERSION":
                        ConfigWarning.ert_context_warn(
                            "IES_INVERSION is deprecated, please use INVERSION instead:\n"
                            f"ANALYSIS_SET_VAR {module_name} INVERSION {new_value.upper()}"
                        )
                    else:
                        ConfigWarning.ert_context_warn(
                            f"Using {value} is deprecated, use:\n"
                            f"ANALYSIS_SET_VAR {module_name} INVERSION {new_value.upper()}"
                        )
                    value = new_value

                var_name = "inversion"
            key = var_name.lower()
            try:
                options[module_name][key] = value
            except KeyError:
                all_errors.append(
                    ConfigValidationError(
                        f"Invalid configuration: ANALYSIS_SET_VAR {module_name} {var_name}"
                    )
                )

        if errors:
            all_errors.append(
                ConfigValidationError(
                    f"The {', '.join(errors)} keyword(s) has been removed and functionality "
                    "replaced with the ENKF_TRUNCATION keyword. Please see "
                    "https://ert.readthedocs.io/en/latest/reference/configuration/keywords.html#enkf-truncation "
                    "for documentation how to use this instead."
                )
            )

        try:
            self.es_module = ESSettings(**options["STD_ENKF"])
            self.ies_module = IESSettings(**options["IES_ENKF"])
            self.observation_settings = UpdateSettings(**observation_settings)
        except ValidationError as err:
            for error in err.errors():
                error["loc"] = tuple(
                    [val.upper() for val in error["loc"] if isinstance(val, str)]
                )
                all_errors.append(ConfigValidationError(str(error)))

        if all_errors:
            raise ConfigValidationError.from_collected(all_errors)

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
            ConfigWarning.ert_context_warn(
                "MIN_REALIZATIONS set to more than NUM_REALIZATIONS, "
                "will set required to successful realizations to 100%. "
                "For more flexibility, you can use e.g. 'MIN_REALIZATIONS 80%'.",
                min_realization_str,
            )

        min_realization = min(min_realization, num_realization)

        config = cls(
            alpha=config_dict.get(ConfigKeys.ENKF_ALPHA, 3.0),
            std_cutoff=config_dict.get(ConfigKeys.STD_CUTOFF, 1e-6),
            max_runtime=config_dict.get(ConfigKeys.MAX_RUNTIME, 0),
            min_realization=min_realization,
            update_log_path=config_dict.get(ConfigKeys.UPDATE_LOG_PATH, "update_log"),
            analysis_iter_config=AnalysisIterConfig(**config_dict),
            analysis_set_var=config_dict.get(ConfigKeys.ANALYSIS_SET_VAR, []),
        )
        return config

    @property
    def log_path(self) -> Path:
        return Path(realpath(self._update_log_path))

    @log_path.setter
    def log_path(self, log_path: Union[str, Path]) -> None:
        self._update_log_path = Path(log_path)

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
    def max_runtime(self) -> Optional[int]:
        return self._max_runtime if self._max_runtime > 0 else None

    def have_enough_realisations(self, realizations: int) -> bool:
        return realizations >= self.minimum_required_realizations

    @property
    def ensemble_format(self) -> Optional[str]:
        return self._analysis_iter_config.iter_case

    def ensemble_format_is_set(self) -> bool:
        return self._analysis_iter_config.iter_case is not None

    def set_ensemble_format(self, ensemble_fmt: str) -> None:
        self._analysis_iter_config.iter_case = ensemble_fmt

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
            f"max_runtime={self._max_runtime}, "
            f"min_realization={self._min_realization}, "
            f"update_log_path={self._update_log_path}, "
            f"analysis_iter_config={self._analysis_iter_config}, "
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnalysisConfig):
            return False

        if self.log_path != other.log_path:
            return False

        if self.max_runtime != other.max_runtime:
            return False

        if self.observation_settings != other.observation_settings:
            return False

        if self.ies_module != other.ies_module:
            return False

        if self.es_module != other.es_module:
            return False

        if self._analysis_iter_config != other._analysis_iter_config:
            return False

        return self.minimum_required_realizations == other.minimum_required_realizations
