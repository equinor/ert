from __future__ import annotations

import math
import warnings
from dataclasses import fields
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError, field_validator, model_validator
from pydantic.dataclasses import dataclass
from scipy.stats import norm

from .parsing import ConfigValidationError


class TransSettingsValidation:
    @classmethod
    def create(cls, *args, **kwargs):
        try:
            return cls(*args, **kwargs)
        except ValidationError as e:
            simplified_msg = "; ".join(err["msg"] for err in e.errors())
            raise ConfigValidationError(simplified_msg) from e

    @classmethod
    def get_param_names(cls) -> list[str]:
        return [f.name for f in fields(cls) if f.name != "name"]


@dataclass
class TransUnifSettings(TransSettingsValidation):
    name: Literal["uniform"] = "uniform"
    min: float = 0.0
    max: float = 1.0

    def trans(self, x: float) -> float:
        y = float(norm.cdf(x))
        return y * (self.max - self.min) + self.min


@dataclass
class TransLogUnifSettings(TransSettingsValidation):
    name: Literal["logunif"] = "logunif"
    min: float = 0.0
    max: float = 1.0

    def trans(self, x: float) -> float:
        log_min, log_max = math.log(self.min), math.log(self.max)
        tmp = norm.cdf(x)
        log_y = log_min + tmp * (log_max - log_min)  # Shift according to max / min
        return math.exp(log_y)


@dataclass
class TransDUnifSettings(TransSettingsValidation):
    name: Literal["dunif"] = "dunif"
    steps: int = 1000
    min: float = 0.0
    max: float = 1.0

    def trans(self, x: float) -> float:
        y = float(norm.cdf(x))
        return (math.floor(y * self.steps) / (self.steps - 1)) * (
            self.max - self.min
        ) + self.min


@dataclass
class TransNormalSettings(TransSettingsValidation):
    name: Literal["normal"] = "normal"
    mean: float = 0.0
    std: float = 1.0

    @field_validator("std")
    @classmethod
    def std_must_be_positive(cls, value):
        if value < 0:
            raise ValueError(f"Negative STD {value} for normal distribution")
        return value

    def trans(self, x: float) -> float:
        return x * self.std + self.mean


@dataclass
class TransLogNormalSettings(TransSettingsValidation):
    name: Literal["lognormal"] = "lognormal"
    mean: float = 0.0
    std: float = 1.0

    @field_validator("std")
    @classmethod
    def std_must_be_positive(cls, value):
        if value < 0:
            raise ValueError(f"Negative STD {value} for lognormal distribution")
        return value

    def trans(self, x: float) -> float:
        # mean is the expectation of log( y )
        return math.exp(x * self.std + self.mean)


@dataclass
class TransTruncNormalSettings(TransSettingsValidation):
    name: Literal["truncated_normal"] = "truncated_normal"
    mean: float = 0.0
    std: float = 1.0
    min: float = 0.0
    max: float = 1.0

    @field_validator("std")
    @classmethod
    def std_must_be_positive(cls, value):
        if value < 0:
            raise ValueError(f"Negative STD {value} for truncated normal distribution")
        return value

    def trans(self, x: float) -> float:
        y = x * self.std + self.mean
        return max(min(y, self.max), self.min)  # clamp


@dataclass
class TransRawSettings(TransSettingsValidation):
    name: Literal["raw"] = "raw"

    def trans(self, x: float) -> float:
        return x


@dataclass
class TransConstSettings(TransSettingsValidation):
    name: Literal["const"] = "const"
    value: float = 0.0

    def trans(self, _: float) -> float:
        return self.value


@dataclass
class TransTriangularSettings(TransSettingsValidation):
    name: Literal["triangular"] = "triangular"
    min: float = 0.0
    mode: float = 0.5
    max: float = 1.0

    @model_validator(mode="after")
    def valid_traingular_params(self):
        if not self.min < self.max:
            raise ValueError(
                f"Minimum {self.min} must be strictly less than the maximum {self.max}"
            )
        if not (self.min <= self.mode <= self.max):
            raise ValueError(
                f"The mode {self.mode} must be between the minimum"
                f" {self.min} and maximum {self.max}"
            )
        return self

    def trans(self, x: float) -> float:
        inv_norm_left = (self.max - self.min) * (self.mode - self.min)
        inv_norm_right = (self.max - self.min) * (self.max - self.mode)
        ymode = (self.mode - self.min) / (self.max - self.min)
        y = norm.cdf(x)

        if y < ymode:
            return self.min + math.sqrt(y * inv_norm_left)
        else:
            return self.max - math.sqrt((1 - y) * inv_norm_right)


@dataclass
class TransErrfSettings(TransSettingsValidation):
    name: Literal["errf"] = "errf"
    min: float = 0.0
    max: float = 1.0
    skewness: float = 0.0
    width: float = 1.0

    def trans(self, x: float) -> float:
        y = norm(loc=0, scale=self.width).cdf(x + self.skewness)
        if np.isnan(y):
            raise ValueError(
                "Output is nan, likely from triplet (x, skewness, width) "
                "leading to low/high-probability in normal CDF."
            )
        return self.min + y * (self.max - self.min)


@dataclass
class TransDerrfSettings(TransSettingsValidation):
    name: Literal["derrf"] = "derrf"
    steps: float = 1000.0
    min: float = 0.0
    max: float = 1.0
    skewness: float = 0.0
    width: float = 1.0

    @model_validator(mode="after")
    def valid_derrf_params(self):
        steps_float = float(self.steps)
        if not steps_float.is_integer() or not (int(steps_float) >= 1):
            raise ValueError(
                f"NBINS {int(self.steps)} must be a positive integer"
                " larger than 1 for DERRF distribution"
            )
        self.steps = int(self.steps)
        if not (self.min < self.max):
            raise ValueError(
                f"The minimum {self.min} must be less than "
                f"the maximum {self.max} for DERRF distribution"
            )
        if not (self.width > 0):
            raise ValueError(
                f"The width {self.width} must be greater than 0 for DERRF distribution"
            )
        return self

    def trans(self, x: float) -> float:
        q_values = np.linspace(start=0, stop=1, num=int(self.steps))
        q_checks = np.linspace(start=0, stop=1, num=int(self.steps + 1))[1:]
        y = TransErrfSettings(
            min=0, max=1, skewness=self.skewness, width=self.width
        ).trans(x)
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


# def get_distribution(name: str, values: list[str]) -> Any:
#     return {
#         "NORMAL": lambda: TransNormalSettings.create(
#             mean=float(values[0]), std=float(values[1])
#         ),
#         "LOGNORMAL": lambda: TransLogNormalSettings.create(
#             mean=float(values[0]), std=float(values[1])
#         ),
#         "UNIFORM": lambda: TransUnifSettings.create(
#             min=float(values[0]), max=float(values[1])
#         ),
#         "LOGUNIF": lambda: TransLogUnifSettings.create(
#             min=float(values[0]), max=float(values[1])
#         ),
#         "TRUNCATED_NORMAL": lambda: TransTruncNormalSettings.create(
#             mean=float(values[0]),
#             std=float(values[1]),
#             min=float(values[2]),
#             max=float(values[3]),
#         ),
#         "RAW": TransRawSettings.create,
#         "CONST": lambda: TransConstSettings.create(value=float(values[0])),
#         "DUNIF": lambda: TransDUnifSettings.create(
#             steps=int(values[0]), min=float(values[1]), max=float(values[2])
#         ),
#         "TRIANGULAR": lambda: TransTriangularSettings.create(
#             min=float(values[0]), mode=float(values[1]), max=float(values[2])
#         ),
#         "ERRF": lambda: TransErrfSettings.create(
#             min=values[0],
#             max=values[1],
#             skewness=values[2],
#             width=values[3],
#         ),
#         "DERRF": lambda: TransDerrfSettings.create(
#             steps=values[0],
#             min=values[1],
#             max=values[2],
#             skewness=values[3],
#             width=values[4],
#         ),
#     }[name]()


DISTRIBUTION_CLASSES = {
    "NORMAL": TransNormalSettings,
    "LOGNORMAL": TransLogNormalSettings,
    "UNIFORM": TransUnifSettings,
    "LOGUNIF": TransLogUnifSettings,
    "TRUNCATED_NORMAL": TransTruncNormalSettings,
    "RAW": TransRawSettings,
    "CONST": TransConstSettings,
    "DUNIF": TransDUnifSettings,
    "TRIANGULAR": TransTriangularSettings,
    "ERRF": TransErrfSettings,
    "DERRF": TransDerrfSettings,
}


def get_distribution(name: str, values: list[str]) -> Any:
    cls = DISTRIBUTION_CLASSES[name]

    param_names = cls.get_param_names()

    # Prepare typed kwargs: try float first, fallback to str
    kwargs = {}
    for pname, val in zip(param_names, values, strict=False):
        try:
            fval = float(val)
            kwargs[pname] = fval
        except ValueError:
            kwargs[pname] = val

    return cls.create(**kwargs)
