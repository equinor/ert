from __future__ import annotations

import math
import warnings
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, field_validator, model_validator
from scipy.stats import norm

from .parsing import ConfigValidationError, ErrorInfo


class TransSettingsValidation(BaseModel):
    model_config = {"extra": "forbid"}

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> TransSettingsValidation:
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

    def transform(self, x: float) -> float:
        y = float(norm.cdf(x))
        return y * (self.max - self.min) + self.min


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

    def transform(self, x: float) -> float:
        log_min, log_max = math.log(self.min), math.log(self.max)
        tmp = norm.cdf(x)
        # Shift according to max / min
        return math.exp(log_min + tmp * (log_max - log_min))


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

    def transform(self, x: float) -> float:
        y = norm.cdf(x)
        return (math.floor(y * self.steps) / (self.steps - 1)) * (
            self.max - self.min
        ) + self.min


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

    def transform(self, x: float) -> float:
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

    def transform(self, x: float) -> float:
        return math.exp(x * self.std + self.mean)


class TruncNormalSettings(TransSettingsValidation):
    name: Literal["truncated_normal"] = "truncated_normal"
    mean: float = 0.0
    std: float = 1.0
    min: float = 0.0
    max: float = 1.0

    @model_validator(mode="after")
    def valid_trunc_normal_params(self) -> Self:
        errors = []
        if not (self.min < self.max):
            errors.append(
                ErrorInfo(
                    message=f"Minimum {self.min} must be strictly less than"
                    f" the maximum {self.max} for truncated_normal distribution"
                )
            )
        if self.std <= 0:
            errors.append(
                ErrorInfo(
                    message=f"Negative STD {self.std} for truncated normal distribution"
                )
            )
        if errors:
            raise ConfigValidationError.from_collected(errors)
        return self

    def transform(self, x: float) -> float:
        y = x * self.std + self.mean
        return max(min(y, self.max), self.min)  # clamp


class RawSettings(TransSettingsValidation):
    name: Literal["raw"] = "raw"

    def transform(self, x: float) -> float:
        return x


class ConstSettings(TransSettingsValidation):
    name: Literal["const"] = "const"
    value: float = 0.0

    def transform(self, _: float) -> float:
        return self.value


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

    def transform(self, x: float) -> float:
        inv_norm_left = (self.max - self.min) * (self.mode - self.min)
        inv_norm_right = (self.max - self.min) * (self.max - self.mode)
        ymode = (self.mode - self.min) / (self.max - self.min)
        y = norm.cdf(x)

        if y < ymode:
            return self.min + math.sqrt(y * inv_norm_left)
        else:
            return self.max - math.sqrt((1 - y) * inv_norm_right)


class ErrfSettings(TransSettingsValidation):
    name: Literal["errf"] = "errf"
    min: float = 0.0
    max: float = 1.0
    skewness: float = 0.0
    width: float = 1.0

    @model_validator(mode="after")
    def valid_errf_params(self) -> Self:
        errors = []
        if not self.width > 0:
            errors.append(
                ErrorInfo(
                    message=f"The width {self.width} must be greater than"
                    " 0 for errf distribution"
                )
            )
        if not (self.min < self.max):
            errors.append(
                ErrorInfo(
                    message=f"Minimum {self.min} must be strictly less"
                    f" than the maximum {self.max} for errf distribution"
                )
            )
        if errors:
            raise ConfigValidationError.from_collected(errors)
        return self

    def transform(self, x: float) -> float:
        y = norm(loc=0, scale=self.width).cdf(x + self.skewness)
        if np.isnan(y):
            raise ValueError(
                "Output is nan, likely from triplet (x, skewness, width) "
                "leading to low/high-probability in normal CDF."
            )
        return self.min + y * (self.max - self.min)


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

    def transform(self, x: float) -> float:
        q_values = np.linspace(start=0, stop=1, num=int(self.steps))
        q_checks = np.linspace(start=0, stop=1, num=int(self.steps + 1))[1:]
        y = ErrfSettings(
            min=0, max=1, skewness=self.skewness, width=self.width
        ).transform(x)
        bin_index = np.digitize(y, q_checks, right=True)
        y_binned = q_values[bin_index]
        result = self.min + y_binned * (self.max - self.min)
        if result > self.max or result < self.min:
            warnings.warn(
                "trans_derff suffered from catastrophic"
                " loss of precision, clamping to min,max",
                stacklevel=1,
            )
            return np.clip(result, self.min, self.max)
        if np.isnan(result):
            raise ValueError(
                "trans_derrf returns nan, check that input arguments are reasonable"
            )
        return float(result)


DISTRIBUTION_CLASSES: dict[str, type[TransSettingsValidation]] = {
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


def get_distribution(name: str, values: list[float]) -> Any:
    cls = DISTRIBUTION_CLASSES[name]

    param_names = cls.get_param_names()

    kwargs = dict(zip(param_names, values, strict=False))

    return cls.create(**kwargs)
