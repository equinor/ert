import math
import os
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self, overload

import numpy as np
from scipy.stats import norm

from ._str_to_bool import str_to_bool
from .parameter_config import parse_config
from .parsing import ConfigValidationError, ConfigWarning


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


@overload
def _get_abs_path(file: None) -> None:
    pass


@overload
def _get_abs_path(file: str) -> str:
    pass


def _get_abs_path(file: str | None) -> str | None:
    if file is not None:
        file = os.path.realpath(file)
    return file


def get_distribution(name: str, values: list[str]) -> Any:
    return {
        "NORMAL": TransNormalSettings(mean=float(values[0]), std=float(values[1])),
        "UNIFORM": TransUnifSettings(min=float(values[0]), max=float(values[1])),
        "TRUNC_NORMAL": TransTruncNormalSettings(
            mean=float(values[0]),
            std=float(values[1]),
            min=float(values[2]),
            max=float(values[3]),
        ),
        "RAW": TransRawSettings(),
        "CONST": TransConstSettings(value=float(values[0])),
        "DUNIF": TransDUnifSettings(
            steps=int(values[0]), min=float(values[1]), max=float(values[2])
        ),
        "TRIANGULAR": TransTriangularSettings(
            min=float(values[0]), mode=float(values[1]), max=float(values[2])
        ),
        "ERRF": TransErrfSettings(
            min=float(values[0]),
            max=float(values[1]),
            skew=float(values[2]),
            width=float(values[3]),
        ),
        "DERRF": TransDerrfSettings(
            steps=int(values[0]),
            min=float(values[1]),
            max=float(values[2]),
            skew=float(values[3]),
            width=float(values[4]),
        ),
    }[name]


class DataSource(StrEnum):
    DESIGN_MATRIX = "design_matrix"
    SAMPLED = "sampled"


@dataclass
class ScalarParameter:
    template_file: str | None
    output_file: str | None
    param_name: str
    group_name: str
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
    # active: bool
    input_source: DataSource
    # dataset_file: PolarsData | None

    @classmethod
    def from_config_list(cls, gen_kw: list[str]) -> list[Self]:
        gen_kw_key = gen_kw[0]

        positional_args, options = parse_config(gen_kw, 4)
        forward_init = str_to_bool(options.get("FORWARD_INIT", "FALSE"))
        init_file = _get_abs_path(options.get("INIT_FILES"))
        update_parameter = str_to_bool(options.get("UPDATE", "TRUE"))
        errors = []

        if len(positional_args) == 2:
            parameter_file = _get_abs_path(positional_args[1])
            parameter_file_context = positional_args[1]
            template_file = None
            output_file = None
        elif len(positional_args) == 4:
            output_file = positional_args[2]
            parameter_file = _get_abs_path(positional_args[3])
            parameter_file_context = positional_args[3]
            template_file = _get_abs_path(positional_args[1])
            if not os.path.isfile(template_file):
                errors.append(
                    ConfigValidationError.with_context(
                        f"No such template file: {template_file}", positional_args[1]
                    )
                )
            elif Path(template_file).stat().st_size == 0:
                token = (
                    parameter_file_context.token
                    if hasattr(parameter_file_context, "token")
                    else parameter_file_context
                )
                ConfigWarning.deprecation_warn(
                    f"The template file for GEN_KW ({gen_kw_key}) is empty. If templating is not needed, you "
                    f"can use GEN_KW with just the distribution file instead: GEN_KW {gen_kw_key} {token}",
                    positional_args[1],
                )

        else:
            raise ConfigValidationError(
                f"Unexpected positional arguments: {positional_args}"
            )
        if not os.path.isfile(parameter_file):
            errors.append(
                ConfigValidationError.with_context(
                    f"No such parameter file: {parameter_file}", parameter_file_context
                )
            )
        elif Path(parameter_file).stat().st_size == 0:
            errors.append(
                ConfigValidationError.with_context(
                    f"No parameters specified in {parameter_file}",
                    parameter_file_context,
                )
            )

        if forward_init:
            errors.append(
                ConfigValidationError.with_context(
                    "Loading GEN_KW from files created by the forward "
                    "model is not supported.",
                    gen_kw,
                )
            )

        if init_file:
            errors.append(
                ConfigValidationError.with_context(
                    "Loading GEN_KW from init_files is not longer supported!",
                    gen_kw,
                )
            )

        if errors:
            raise ConfigValidationError.from_collected(errors)

        parameter_configuration: list[Self] = []
        with open(parameter_file, encoding="utf-8") as file:
            for line_number, item in enumerate(file):
                item = item.split("--")[0]  # remove comments
                if item.strip():  # only lines with content
                    items = item.split()
                    if len(items) < 2:
                        errors.append(
                            ConfigValidationError.with_context(
                                f"Too few values on line {line_number} in parameter file {parameter_file}",
                                gen_kw,
                            )
                        )
                    else:
                        parameter_configuration.append(
                            cls(
                                param_name=items[1],
                                input_source=DataSource.SAMPLED,
                                group_name=gen_kw_key,
                                distribution=get_distribution(items[0], items[2:]),
                                template_file=template_file,
                                output_file=output_file,
                            )
                        )

        if errors:
            raise ConfigValidationError.from_collected(errors)

        if gen_kw_key == "PRED" and update_parameter:
            ConfigWarning.warn(
                "GEN_KW PRED used to hold a special meaning and be "
                "excluded from being updated.\n If the intention was "
                "to exclude this from updates, set UPDATE:FALSE.\n",
                gen_kw[0],
            )
        return parameter_configuration

        # return cls(
        #     name=gen_kw_key,
        #     forward_init=forward_init,
        #     template_file=template_file,
        #     output_file=output_file,
        #     forward_init_file=init_file,
        #     transform_function_definitions=transform_function_definitions,
        #     update=update_parameter,
        # )
