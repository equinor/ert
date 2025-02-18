from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import xarray as xr
from scipy.stats import norm

from .parameter_config import ParameterConfig

# parameter_configs = List[FieldParameter, SurfaceParameter, ScalarParameter]


@dataclass
class TransUniformfSettings:
    name: Literal["uniform"] = "uniform"
    min: float = 0.0
    max: float = 1.0

    def trans(self, x: float) -> float:
        y = float(norm.cdf(x))
        return y * (self.max - self.min) + self.min


@dataclass
class TransNormalfSettings:
    name: Literal["normal"] = "normal"
    mean: float = 0.0
    std: float = 1.0

    def trans(self, x: float) -> float:
        return x * self.std + self.mean


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
class PolarsData:
    name: Literal["polars"]
    data_set_file: Path


@dataclass
class ScalarParameter(ParameterConfig):
    name: str
    group: str
    distribution: (
        TransUniformfSettings
        | TransRawSettings
        | TransConstSettings
        | TransNormalfSettings
    )
    active: bool
    input_source: Literal["design_matrix", "sampled"]
    dataset_file: PolarsData | xr.Dataset
