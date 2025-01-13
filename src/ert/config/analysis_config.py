from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import ceil
from os.path import realpath
from pathlib import Path
from typing import Any, Final, no_type_check

from pydantic import PositiveFloat, ValidationError

from .analysis_module import ESSettings, IESSettings
from .design_matrix import DesignMatrix
from .parsing import (
    AnalysisMode,
    ConfigDict,
    ConfigKeys,
    ConfigValidationError,
    ConfigWarning,
)

logger = logging.getLogger(__name__)

DEFAULT_ANALYSIS_MODE = AnalysisMode.ENSEMBLE_SMOOTHER
ObservationGroups = list[str]


@dataclass
class UpdateSettings:
    std_cutoff: PositiveFloat = 1e-6
    alpha: float = 3.0
    auto_scale_observations: list[ObservationGroups] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    minimum_required_realizations: int = 0
    update_log_path: str | Path = "update_log"
    es_module: ESSettings = field(default_factory=ESSettings)
    ies_module: IESSettings = field(default_factory=IESSettings)
    observation_settings: UpdateSettings = field(default_factory=UpdateSettings)
    num_iterations: int = 1
    design_matrix: DesignMatrix | None = None

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> AnalysisConfig:
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
            ConfigWarning.warn(
                "MIN_REALIZATIONS set to more than NUM_REALIZATIONS, "
                "will set required to successful realizations to 100%. "
                "For more flexibility, you can use e.g. 'MIN_REALIZATIONS 80%'.",
                min_realization_str,
            )

        min_realization = min(min_realization, num_realization)

        design_matrix_config_lists = config_dict.get(ConfigKeys.DESIGN_MATRIX, [])

        options: dict[str, dict[str, Any]] = {"STD_ENKF": {}, "IES_ENKF": {}}
        observation_settings: dict[str, Any] = {
            "alpha": config_dict.get(ConfigKeys.ENKF_ALPHA, 3.0),
            "std_cutoff": config_dict.get(ConfigKeys.STD_CUTOFF, 1e-6),
            "auto_scale_observations": [],
        }
        analysis_set_var = config_dict.get(ConfigKeys.ANALYSIS_SET_VAR, [])
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
            if var_name in {"INVERSION", "IES_INVERSION"}:
                if value in inversion_str_map[module_name]:
                    new_value = inversion_str_map[module_name][value]
                    if var_name == "IES_INVERSION":
                        ConfigWarning.warn(
                            "IES_INVERSION is deprecated, please use INVERSION instead:\n"
                            f"ANALYSIS_SET_VAR {module_name} INVERSION {new_value.upper()}"
                        )
                    else:
                        ConfigWarning.warn(
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
            es_settings = ESSettings(**options["STD_ENKF"])
            ies_settings = IESSettings(**options["IES_ENKF"])
            obs_settings = UpdateSettings(**observation_settings)
        except ValidationError as err:
            for error in err.errors():
                error["loc"] = tuple(
                    val.upper() for val in error["loc"] if isinstance(val, str)
                )
                all_errors.append(ConfigValidationError(str(error)))

        if all_errors:
            raise ConfigValidationError.from_collected(all_errors)

        design_matrices = [
            DesignMatrix.from_config_list(design_matrix_config_list)
            for design_matrix_config_list in design_matrix_config_lists
        ]
        design_matrix: DesignMatrix | None = None
        if design_matrices:
            design_matrix = design_matrices[0]
            for dm_other in design_matrices[1:]:
                design_matrix.merge_with_other(dm_other)
        config = cls(
            minimum_required_realizations=min_realization,
            update_log_path=config_dict.get(ConfigKeys.UPDATE_LOG_PATH, "update_log"),
            observation_settings=obs_settings,
            es_module=es_settings,
            ies_module=ies_settings,
            design_matrix=design_matrix,
        )
        return config

    @property
    def log_path(self) -> Path:
        return Path(realpath(self.update_log_path))

    def __repr__(self) -> str:
        return (
            "AnalysisConfig("
            f"min_realization={self.minimum_required_realizations}, "
            f"update_log_path={self.update_log_path}, "
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnalysisConfig):
            return False

        if self.log_path != other.log_path:
            return False

        if self.observation_settings != other.observation_settings:
            return False

        if self.ies_module != other.ies_module:
            return False

        if self.es_module != other.es_module:
            return False

        return self.minimum_required_realizations == other.minimum_required_realizations
