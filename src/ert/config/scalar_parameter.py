import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import norm

from .parameter_config import ParameterConfig

# parameter_configs = List[FieldParameter, SurfaceParameter, ScalarParameter]


@dataclass
class TransUnifSettings:
    name: Literal["unif"] = "unif"
    min: float = 0.0
    max: float = 1.0

    def trans(self, x: float) -> float:
        y = float(norm.cdf(x))
        return y * (self.max - self.min) + self.min


@dataclass
class TransDUnifSettings:
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
class TransNormalSettings:
    name: Literal["normal"] = "normal"
    mean: float = 0.0
    std: float = 1.0

    def trans(self, x: float) -> float:
        return x * self.std + self.mean


@dataclass
class TransTruncNormalSettings:
    name: Literal["trunc_normal"] = "trunc_normal"
    mean: float = 0.0
    std: float = 1.0
    min: float = 0.0
    max: float = 1.0

    def trans(self, x: float) -> float:
        y = x * self.std + self.mean
        return max(min(y, self.max), self.min)  # clamp


@dataclass
class TransRawSettings:
    name: Literal["raw"] = "raw"

    def trans(self, x: float) -> float:
        return x


@dataclass
class TransConstSettings:
    name: Literal["const"] = "const"
    value: float = 0.0

    def trans(self, _: float) -> float:
        return self.value


@dataclass
class TransTriangularSettings:
    name: Literal["triangular"] = "triangular"
    min: float = 0.0
    mode: float = 0.5
    max: float = 1.0

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
class TransErrfSettings:
    name: Literal["errf"] = "errf"
    min: float = 0.0
    max: float = 1.0
    skew: float = 0.0
    width: float = 1.0

    def trans(self, x: float) -> float:
        y = norm(loc=0, scale=self.width).cdf(x + self.skew)
        if np.isnan(y):
            raise ValueError(
                "Output is nan, likely from triplet (x, skewness, width) "
                "leading to low/high-probability in normal CDF."
            )
        return self.min + y * (self.max - self.min)


@dataclass
class TransDerrfSettings:
    name: Literal["derrf"] = "derrf"
    steps: int = 1000
    min: float = 0.0
    max: float = 1.0
    skew: float = 0.0
    width: float = 1.0

    def trans(self, x: float) -> float:
        q_values = np.linspace(start=0, stop=1, num=self.steps)
        q_checks = np.linspace(start=0, stop=1, num=self.steps + 1)[1:]
        y = TransErrfSettings(min=0, max=1, skew=self.skew, width=self.width).trans(x)
        bin_index = np.digitize(y, q_checks, right=True)
        y_binned = q_values[bin_index]
        result = self.min + y_binned * (self.max - self.min)
        if result > self.max or result < self.min:
            warnings.warn(
                "trans_derff suffered from catastrophic loss of precision, clamping to min,max",
                stacklevel=1,
            )
            return np.clip(result, self.min, self.max)
        if np.isnan(result):
            raise ValueError(
                "trans_derrf returns nan, check that input arguments are reasonable"
            )
        return float(result)


@dataclass
class PolarsData:
    name: Literal["polars"]
    data_set_file: Path


@dataclass
class ScalarParameter(ParameterConfig):
    # name: str
    group: str
    distribution: (
        TransUnifSettings
        | TransDUnifSettings
        | TransRawSettings
        | TransConstSettings
        | TransNormalSettings
        | TransTruncNormalSettings
        | TransErrfSettings
        | TransDerrfSettings
        | TransTriangularSettings
    )
    active: bool
    input_source: Literal["design_matrix", "sampled"]
    dataset_file: PolarsData
