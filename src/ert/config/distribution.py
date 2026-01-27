from __future__ import annotations

import math
import warnings
from typing import Annotated, Any, Literal, Self, TypeVar

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from scipy.special import ndtr

from .parsing import ConfigValidationError, ConfigWarning, ErrorInfo

T = TypeVar("T", bound="TransSettingsValidation")


class TransSettingsValidation(BaseModel):
    model_config = {"extra": "forbid"}

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> Self:
        return cls(*args, **kwargs)

    @classmethod
    def get_param_names(cls) -> list[str]:
        return [
            name
            for name, field in cls.model_fields.items()
            if field.is_required() or (field.default is not None and name != "name")
        ]


class UnifSettings(TransSettingsValidation):
    name: Literal["uniform"] = "uniform"
    min: float = 0.0
    max: float = 1.0

    @model_validator(mode="after")
    def valid_unif_params(self) -> Self:
        if not (self.min < self.max):
            raise ConfigValidationError(
                f"Minimum {self.min} must be strictly less than the maximum {self.max}"
                " for uniform distribution"
            )
        return self

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        y = ndtr(x)
        span = self.max - self.min
        return y * span + self.min


class LogUnifSettings(TransSettingsValidation):
    name: Literal["logunif"] = "logunif"
    min: float = 0.0
    max: float = 1.0

    @model_validator(mode="after")
    def valid_logunif_params(self) -> Self:
        if not (self.min < self.max):
            raise ConfigValidationError(
                f"Minimum {self.min} must be strictly less than the maximum {self.max}"
                " for log uniform distribution"
            )
        return self

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        y = ndtr(x)
        log_min = np.log(self.min)
        log_max = np.log(self.max)
        return np.exp(log_min + y * (log_max - log_min))


class DUnifSettings(TransSettingsValidation):
    name: Literal["dunif"] = "dunif"
    steps: int = 1000
    min: float = 0.0
    max: float = 1.0

    @model_validator(mode="after")
    def valid_dunif_params(self) -> Self:
        errors = []
        if not (self.min < self.max):
            errors.append(
                ErrorInfo(
                    message=f"Minimum {self.min} must be strictly less"
                    f" than the maximum {self.max} for duniform distribution"
                )
            )
        if not self.steps > 1:
            errors.append(
                ErrorInfo(
                    message=f"Number of steps {self.steps} must be larger"
                    " than 1 for duniform distribution"
                )
            )
        if errors:
            raise ConfigValidationError.from_collected(errors)
        return self

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        y = ndtr(x)
        span = self.max - self.min
        steps_denom = float(self.steps - 1)
        return (np.floor(y * self.steps) / steps_denom) * span + self.min


class NormalSettings(TransSettingsValidation):
    name: Literal["normal"] = "normal"
    mean: float = 0.0
    std: float = 1.0

    @field_validator("std")
    @classmethod
    def std_must_be_positive(cls, value: float) -> float:
        if value < 0:
            raise ConfigValidationError(f"Negative STD {value} for normal distribution")
        return value

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class LogNormalSettings(TransSettingsValidation):
    name: Literal["lognormal"] = "lognormal"
    mean: float = 0.0
    std: float = 1.0

    @field_validator("std")
    @classmethod
    def std_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ConfigValidationError(
                f"Negative STD {value} for lognormal distribution"
            )
        return value

    @model_validator(mode="after")
    def valid_transformed(self) -> Self:
        # If expected value exp(mu + 0.5 sigma^2) is too large, warn the user
        # 100000 is chosen from input from expert when considering permeability
        # If the expected value overflows, issue an error, as this is definitely wrong
        try:
            expected_value = math.exp(self.mean + 0.5 * self.std**2)
        except OverflowError as e:
            raise ConfigValidationError(
                "Expectation value of the lognormal distribution is too large! "
                "These are the specified parameters:\n\n"
                f"mean (of log values): {self.mean}\n"
                f"standard deviation (of log values): {self.std}\n\n"
                "Remember that the input parameters for keyword "
                "LOGNORMAL are the mean and standard deviation of the natural "
                "logarithm of your parameter data.",
            ) from e

        if expected_value > 100000:
            ConfigWarning.warn(
                "Expectation value of the lognormal distribution is"
                f" {expected_value:.2e}. "
                "This is quite large and may be unintended. "
                "These are the specified parameters:\n\n"
                f"mean (of log values): {self.mean}\n"
                f"standard deviation (of log values): {self.std}\n\n"
                "Remember that the input parameters for keyword "
                "LOGNORMAL are the mean and standard deviation of the natural "
                "logarithm of your parameter data.",
            )
        return self

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x * self.std + self.mean)


class TruncNormalSettings(TransSettingsValidation):
    name: Literal["truncated_normal"] = "truncated_normal"
    mean: float = 0.0
    std: float = 1.0
    min: float = 0.0
    max: float = 1.0

    @field_validator("std")
    @classmethod
    def std_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ConfigValidationError(
                f"Negative STD {value} for truncated normal distribution"
            )
        return value

    @model_validator(mode="after")
    def valid_min_max_relationship(self) -> Self:
        if not (self.min < self.max):
            raise ConfigValidationError(
                f"Minimum {self.min} must be strictly less than"
                f" the maximum {self.max} for truncated_normal distribution"
            )
        return self

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x * self.std + self.mean, self.min, self.max)


class RawSettings(TransSettingsValidation):
    name: Literal["raw"] = "raw"

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x


class ConstSettings(TransSettingsValidation):
    name: Literal["const"] = "const"
    value: float = 0.0

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(x, self.value, dtype=np.float64)


class TriangularSettings(TransSettingsValidation):
    name: Literal["triangular"] = "triangular"
    min: float = 0.0
    mode: float = 0.5
    max: float = 1.0

    @model_validator(mode="after")
    def valid_triangular_params(self) -> Self:
        errors = []
        if not self.min < self.max:
            errors.append(
                ErrorInfo(
                    message=f"Minimum {self.min} must be strictly less than"
                    f" the maximum {self.max} for triangular distribution"
                )
            )
        if not (self.min <= self.mode <= self.max):
            errors.append(
                ErrorInfo(
                    message=f"The mode {self.mode} must be between the minimum"
                    f" {self.min} and maximum {self.max} for triangular distribution"
                )
            )
        if errors:
            raise ConfigValidationError.from_collected(errors)
        return self

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        y = ndtr(x)

        span = self.max - self.min
        ymode = (self.mode - self.min) / span

        left_mask = y < ymode

        result = np.empty_like(y, dtype=np.float64)
        if np.any(left_mask):
            y_left = y[left_mask]
            inv_norm_left = span * (self.mode - self.min)
            result[left_mask] = self.min + np.sqrt(y_left * inv_norm_left)
        if np.any(~left_mask):
            y_right = y[~left_mask]
            inv_norm_right = span * (self.max - self.mode)
            result[~left_mask] = self.max - np.sqrt((1.0 - y_right) * inv_norm_right)
        return result


class ErrfSettings(TransSettingsValidation):
    name: Literal["errf"] = "errf"
    min: float = 0.0
    max: float = 1.0
    skewness: float = 0.0
    width: float = 1.0

    @field_validator("width")
    @classmethod
    def width_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ConfigValidationError(
                f"The width {value} must be greater than 0 for errf distribution"
            )
        return value

    @model_validator(mode="after")
    def valid_min_max_relationship(self) -> Self:
        if not (self.min < self.max):
            raise ConfigValidationError(
                f"Minimum {self.min} must be strictly less than"
                f" the maximum {self.max} for errf distribution"
            )
        return self

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        # CDF of N(0, width) at (x + skewness)
        span = self.max - self.min
        inv_width = 1.0 / self.width
        y = ndtr((x + self.skewness) * inv_width)
        if np.any(np.isnan(y)):
            raise ValueError(
                "Output is nan, likely from triplet (x, skewness, width) "
                "leading to low/high-probability in normal CDF."
            )
        return self.min + y * span


class DerrfSettings(TransSettingsValidation):
    name: Literal["derrf"] = "derrf"
    steps: float = 1000.0
    min: float = 0.0
    max: float = 1.0
    skewness: float = 0.0
    width: float = 1.0

    @model_validator(mode="after")
    def valid_derrf_params(self) -> Self:
        errors = []
        steps_float = float(self.steps)
        if not steps_float.is_integer() or not (int(steps_float) >= 1):
            errors.append(
                ErrorInfo(
                    message=f"NBINS {self.steps} must be a positive integer"
                    " larger than 1 for DERRF distributed"
                )
            )
        self.steps = int(self.steps)
        if not (self.min < self.max):
            errors.append(
                ErrorInfo(
                    message=f"The minimum {self.min} must be less than "
                    f"the maximum {self.max} for DERRF distributed"
                )
            )
        if not (self.width > 0):
            errors.append(
                ErrorInfo(
                    message=f"The width {self.width} must be greater"
                    " than 0 for DERRF distributed"
                )
            )
        if errors:
            raise ConfigValidationError.from_collected(errors)
        return self

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        # Vectorized equivalent of DerrfSettings.transform
        span = self.max - self.min
        inv_width = 1.0 / self.width
        steps_int = int(self.steps)
        q_values = np.linspace(start=0.0, stop=1.0, num=steps_int)
        q_checks = np.linspace(start=0.0, stop=1.0, num=steps_int + 1)[1:]

        y = ndtr((x + self.skewness) * inv_width)
        if np.any(np.isnan(y)):
            raise ValueError(
                "trans_derrf returns nan, check that input arguments are reasonable"
            )

        # Equivalent to np.digitize(y, q_checks, right=True) for y in [0,1]
        bin_index = np.searchsorted(q_checks, y, side="left")
        y_binned = q_values[bin_index]

        result = self.min + y_binned * span
        if np.any((result > self.max) | (result < self.min)):
            warnings.warn(
                (
                    "trans_derff suffered from catastrophic loss"
                    "of precision, clamping to min,max"
                ),
                stacklevel=1,
            )
        return np.clip(result, self.min, self.max)


DistributionSettings = Annotated[
    UnifSettings
    | LogNormalSettings
    | LogUnifSettings
    | DUnifSettings
    | RawSettings
    | ConstSettings
    | NormalSettings
    | TruncNormalSettings
    | ErrfSettings
    | DerrfSettings
    | TriangularSettings,
    Field(discriminator="name"),
]

DISTRIBUTION_CLASSES: dict[str, type[DistributionSettings]] = {
    "NORMAL": NormalSettings,
    "LOGNORMAL": LogNormalSettings,
    "UNIFORM": UnifSettings,
    "LOGUNIF": LogUnifSettings,
    "TRUNCATED_NORMAL": TruncNormalSettings,
    "RAW": RawSettings,
    "CONST": ConstSettings,
    "DUNIF": DUnifSettings,
    "TRIANGULAR": TriangularSettings,
    "ERRF": ErrfSettings,
    "DERRF": DerrfSettings,
}


def get_distribution(name: str, values: list[float]) -> DistributionSettings:
    cls = DISTRIBUTION_CLASSES[name]

    param_names = cls.get_param_names()
    kwargs = dict(zip(param_names, values, strict=False))

    return cls.create(**kwargs)
