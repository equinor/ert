from __future__ import annotations

import dataclasses
import logging
from dataclasses import field
from math import ceil
from os.path import realpath
from pathlib import Path
from typing import Any, Final, no_type_check

from pydantic import Field, PositiveFloat, ValidationError
from pydantic.dataclasses import dataclass

from .analysis_module import ESSettings
from .design_matrix import DesignMatrix
from .parsing import (
    ConfigDict,
    ConfigKeys,
    ConfigValidationError,
    ConfigWarning,
)

logger = logging.getLogger(__name__)

ObservationGroups = list[str]


@dataclass
class OutlierSettings:
    alpha: float = Field(default=3.0)
    std_cutoff: PositiveFloat = Field(default=1e-6)


@dataclass
class ObservationSettings:
    outlier_settings: OutlierSettings = Field(default_factory=OutlierSettings)
    auto_scale_observations: list[ObservationGroups] | None = Field(
        default_factory=list
    )


@dataclasses.dataclass
class AnalysisConfig:
    minimum_required_realizations: int = 0
    update_log_path: str | Path = "update_log"
    es_settings: ESSettings = field(default_factory=ESSettings)
    observation_settings: ObservationSettings = field(
        default_factory=ObservationSettings
    )
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
                "MIN_REALIZATIONS is set to more than NUM_REALIZATIONS. "
                "Will continue with required number of successful "
                "realizations set to NUM_REALIZATIONS. "
                "For more flexibility, you can use e.g. 'MIN_REALIZATIONS 80%'.",
                min_realization_str,
            )

        min_realization = min(min_realization, num_realization)

        design_matrix_config_lists = config_dict.get(ConfigKeys.DESIGN_MATRIX, [])

        options: dict[str, dict[str, Any]] = {"STD_ENKF": {}}

        auto_scale_observations: list[str] = []
        analysis_set_var = config_dict.get(ConfigKeys.ANALYSIS_SET_VAR, [])
        inversion_str_map: Final = {
            "STD_ENKF": {
                **dict.fromkeys(["exact", "0"], "EXACT"),
                **dict.fromkeys(
                    ["subspace", "subspace_exact_r", "SUBSPACE_EXACT_R", "1"],
                    "SUBSPACE",
                ),
            }
        }
        deprecated_keys = ["ENKF_NCOMP", "ENKF_SUBSPACE_DIMENSION"]
        deprecated_inversion_keys = ["USE_EE", "USE_GE"]
        errors = []
        all_errors = []

        for module_name, var_name, value in analysis_set_var:
            if module_name == "IES_ENKF":
                ConfigWarning.warn(
                    f"{module_name} has been removed and has no effect, valid "
                    "options are:\nANALYSIS_SET_VAR STD_ENKF ..."
                )
                continue
            if module_name == "OBSERVATIONS":
                if var_name == "AUTO_SCALE":
                    auto_scale_observations.append(value.split(","))
                else:
                    all_errors.append(
                        ConfigValidationError(
                            f"Unknown variable: {var_name} for: ANALYSIS_SET_VAR "
                            f"OBSERVATIONS {var_name}\nValid options: AUTO_SCALE"
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
                        f"Keyword {var_name} has been replaced by INVERSION and "
                        "has no effect.\n\nPlease see "
                        "https://ert.readthedocs.io/en/latest/reference/configuration/keywords.html#inversion-algorithm "  # noqa: E501
                        "for documentation how to use this instead."
                    )
                )
                continue
            if var_name == "INVERSION":
                if value in inversion_str_map[module_name]:
                    new_value = inversion_str_map[module_name][value]
                    ConfigWarning.warn(
                        f"Using {value} is deprecated, use:\n"
                        f"ANALYSIS_SET_VAR {module_name} INVERSION {new_value}"
                    )
                    value = new_value

                var_name = "inversion"
            key = var_name.lower()
            try:
                options[module_name][key] = value
            except KeyError:
                all_errors.append(
                    ConfigValidationError(
                        "Invalid configuration: ANALYSIS_SET_VAR "
                        f"{module_name} {var_name}"
                    )
                )

        if errors:
            all_errors.append(
                ConfigValidationError(
                    f"The {', '.join(errors)} keyword(s) has been removed and "
                    "functionality replaced with the ENKF_TRUNCATION keyword. "
                    "Please see "
                    "https://ert.readthedocs.io/en/latest/reference/configuration/keywords.html#enkf-truncation "  # noqa: E501
                    "for documentation how to use this instead."
                )
            )

        try:
            es_settings = ESSettings(**options["STD_ENKF"])
            outlier_settings: dict[str, Any] = {
                "alpha": config_dict.get(ConfigKeys.ENKF_ALPHA, 3.0),
                "std_cutoff": config_dict.get(ConfigKeys.STD_CUTOFF, 1e-6),
            }

            obs_settings = ObservationSettings(
                outlier_settings=OutlierSettings(**outlier_settings),
                auto_scale_observations=auto_scale_observations,
            )
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
                if design_matrix != dm_other:
                    design_matrix.merge_with_other(dm_other)
                else:
                    logger.warning(
                        f"Duplicate DESIGN_MATRIX entries {dm_other}, "
                        "only reading once."
                    )
            logger.info("Running ERT with DESIGN_MATRIX")
        config = cls(
            minimum_required_realizations=min_realization,
            update_log_path=config_dict.get(ConfigKeys.UPDATE_LOG_PATH, "update_log"),
            observation_settings=obs_settings,
            es_settings=es_settings,
            design_matrix=design_matrix,
        )
        return config

    @property
    def log_path(self) -> Path:
        return Path(realpath(self.update_log_path))
