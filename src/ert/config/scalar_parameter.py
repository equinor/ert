from __future__ import annotations

import math
import os
import shutil
import warnings
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, overload

import numpy as np
import pandas as pd
import polars as pl
from pydantic.dataclasses import dataclass
from scipy.stats import norm

from ert.substitutions import substitute_runpath_name

from ._str_to_bool import str_to_bool
from .parameter_config import ParameterConfig, parse_config
from .parsing import ConfigValidationError, ConfigWarning

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble


@dataclass
class TransUnifSettings:
    name: Literal["unif"] = "unif"
    min: float = 0.0
    max: float = 1.0

    def trans(self, x: float) -> float:
        y = float(norm.cdf(x))
        return y * (self.max - self.min) + self.min


@dataclass
class TransLogUnifSettings:
    name: Literal["logunif"] = "logunif"
    log_min: float = 0.0
    log_max: float = 1.0

    def trans(self, x: float) -> float:
        # log_min, log_max = math.log(arg[0]), math.log(arg[1])
        tmp = norm.cdf(x)
        log_y = self.log_min + tmp * (
            self.log_max - self.log_min
        )  # Shift according to max / min
        return math.exp(log_y)


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
class TransLogNormalSettings:
    name: Literal["lognormal"] = "lognormal"
    mean: float = 0.0
    std: float = 1.0

    def trans(self, x: float) -> float:
        # mean is the expectation of log( y )
        return math.exp(x * self.std + self.mean)


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
        "NORMAL": lambda: TransNormalSettings(
            mean=float(values[0]), std=float(values[1])
        ),
        "LOGNORMAL": lambda: TransLogNormalSettings(
            mean=float(values[0]), std=float(values[1])
        ),
        "UNIFORM": lambda: TransUnifSettings(
            min=float(values[0]), max=float(values[1])
        ),
        "LOGUNIF": lambda: TransLogUnifSettings(
            log_min=math.log(float(values[0])), log_max=math.log(float(values[1]))
        ),
        "TRUNCATED_NORMAL": lambda: TransTruncNormalSettings(
            mean=float(values[0]),
            std=float(values[1]),
            min=float(values[2]),
            max=float(values[3]),
        ),
        "RAW": TransRawSettings(),
        "CONST": lambda: TransConstSettings(value=float(values[0])),
        "DUNIF": lambda: TransDUnifSettings(
            steps=int(values[0]), min=float(values[1]), max=float(values[2])
        ),
        "TRIANGULAR": lambda: TransTriangularSettings(
            min=float(values[0]), mode=float(values[1]), max=float(values[2])
        ),
        "ERRF": lambda: TransErrfSettings(
            min=float(values[0]),
            max=float(values[1]),
            skew=float(values[2]),
            width=float(values[3]),
        ),
        "DERRF": lambda: TransDerrfSettings(
            steps=int(values[0]),
            min=float(values[1]),
            max=float(values[2]),
            skew=float(values[3]),
            width=float(values[4]),
        ),
    }[name]()


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
        | TransLogNormalSettings
        | TransLogUnifSettings
        | TransDUnifSettings
        | TransRawSettings
        | TransConstSettings
        | TransNormalSettings
        | TransTruncNormalSettings
        | TransErrfSettings
        | TransDerrfSettings
        | TransTriangularSettings
    )
    input_source: DataSource
    update: bool = True


SCALAR_PARAMETERS_NAME = "SCALAR_PARAMETERS"


@dataclass
class ScalarParameters(ParameterConfig):
    scalars: list[ScalarParameter]

    def __post_init__(self) -> None:
        self.groups: dict[str, list[ScalarParameter]] = defaultdict(list)
        self.hash_group_key: dict[str, ScalarParameter] = {}
        for param in self.scalars:
            self.groups[param.group_name].append(param)
            self.hash_group_key[f"{param.group_name}:{param.param_name}"] = param
        self.update = not all(not param.update for param in self.scalars)

    def __getitem__(self, key: str) -> list[ScalarParameter]:
        if key in self.groups:
            return list(self.groups[key])
        elif key in self.hash_group_key:
            return [self.hash_group_key[key]]
        return []

    @staticmethod
    def _sample_value(
        parameters: list[ScalarParameter],
        global_seed: str,
        realization: int,
    ) -> dict[str, float]:
        """
        Generate a sample value for each key in a parameter group.

        The sampling is reproducible and dependent on a global seed combined
        with the parameter group name and individual key names. The 'realization' parameter
        determines the specific sample point from the distribution for each parameter.

        Parameters:
        - parameters (list[ScalarParameter]): List of ScalarParameter.
        The name of the parameter group, used to ensure unique RNG seeds for different groups.
        - global_seed (str): A global seed string used for RNG seed generation to ensure
        reproducibility across runs.
        - realization (int): An integer used to advance the RNG to a specific point in its
        sequence, effectively selecting the 'realization'-th sample from the distribution.

        Returns:
        - dict[str, float]: A dict of sampled values [key:value]

        Note:
        The method uses SHA-256 for hash generation and numpy's default random number generator
        for sampling. The RNG state is advanced to the 'realization' point before generating
        a single sample, enhancing efficiency by avoiding the generation of large, unused sample sets.
        """
        parameter_values: dict[str, float] = {}
        for parameter in parameters:
            if parameter.input_source == DataSource.DESIGN_MATRIX:
                continue
            key_hash = sha256(
                global_seed.encode("utf-8")
                + f"{parameter.group_name}:{parameter.param_name}".encode()
            )
            seed = np.frombuffer(key_hash.digest(), dtype="uint32")
            rng = np.random.default_rng(seed)

            # Advance the RNG state to the realization point
            rng.standard_normal(realization)

            # Generate a single sample
            value = rng.standard_normal(1)
            transformed_value = parameter.distribution.trans(value[0])
            parameter_values[f"{parameter.group_name}:{parameter.param_name}"] = value[
                0
            ]
            parameter_values[
                f"{parameter.group_name}:{parameter.param_name}.transformed"
            ] = float(transformed_value)

        return parameter_values

    def sample_or_load(
        self,
        real_nr: int | Iterable[int],
        random_seed: int,
        ensemble_size: int,
        design_matrix_df: pd.DataFrame | None = None,
    ) -> pl.DataFrame:
        if isinstance(real_nr, int | np.integer):
            real_nr = [real_nr]
        df_list = []
        for real in real_nr:
            params = self._sample_value(
                self.scalars,
                str(random_seed),
                real,
            )
            params["realization"] = real
            if design_matrix_df is not None:
                row = design_matrix_df.loc[real]
                for parameter in self.scalars:
                    if parameter.input_source == DataSource.DESIGN_MATRIX:
                        value = row[parameter.param_name]
                        params[f"{parameter.group_name}:{parameter.param_name}"] = value
                        params[
                            f"{parameter.group_name}:{parameter.param_name}.transformed"
                        ] = value
            df_list.append(pl.DataFrame(params))
        return pl.concat(df_list, how="vertical")

    def load_parameters_to_update(
        self,
        ensemble: Ensemble,
        iens_active_index: npt.NDArray[np.int_],
    ) -> npt.NDArray[np.float64]:
        params_to_update = [
            f"{param.group_name}:{param.param_name}"
            for param in self.scalars
            if param.update
        ]
        if not params_to_update:
            raise ValueError("No parameters to update")
        df = self.load_parameters(ensemble, SCALAR_PARAMETERS_NAME, iens_active_index)
        return df.select(params_to_update).to_numpy().T

    @staticmethod
    def load_parameters(
        ensemble: Ensemble, group: str, realizations: npt.NDArray[np.int_]
    ) -> pl.DataFrame:
        return ensemble.load_parameters_scalar(group, realizations)

    def save_parameters(
        self,
        ensemble: Ensemble,
        group: str,
        realization: int,
        data: npt.NDArray[np.float64],
    ) -> None:
        # this function is not be used since we can't deduce
        # which parameters are being written whithin the data array
        pass

    def save_experiment_data(
        self,
        experiment_path: Path,
    ) -> None:
        templates: dict[str, str] = {}
        for group_name, params in self.groups.items():
            for param in params:
                if param.template_file is not None:
                    templates[group_name] = param.template_file
                    break

        for template_file in templates.values():
            incoming_template_file_path = Path(template_file)
            template_file_path = Path(
                experiment_path / incoming_template_file_path.name
            )
            shutil.copyfile(incoming_template_file_path, template_file_path)

    def save_updated_parameters_and_copy_remaining(
        self,
        source_ensemble: Ensemble,
        target_ensemble: Ensemble,
        group: str,
        realizations: npt.NDArray[np.int_],
        data: npt.NDArray[np.float64],
    ) -> None:
        params_to_update = [
            f"{param.group_name}:{param.param_name}"
            for param in self.scalars
            if param.update
        ]
        df = source_ensemble.load_parameters_scalar(
            scalar_name=group, realizations=realizations
        )
        df_updates = pl.DataFrame(
            {
                "realization": realizations,
                **{col: data[i, :] for i, col in enumerate(params_to_update)},
                **{
                    f"{col}.transformed": [
                        self.hash_group_key[col].distribution.trans(v)
                        for v in data[i, :]
                    ]
                    for i, col in enumerate(params_to_update)
                },
            }
        )
        df = df.update(df_updates, on="realization")
        target_ensemble.save_parameters_scalar(group, realizations, df)

    def __len__(self) -> int:
        return len(self.scalars)

    def read_from_runpath(
        self,
        run_path: Path,
        real_nr: int,
        iteration: int,
    ) -> None:
        """
        forward_init will not be supported, so None for the moment
        """
        return None

    def should_use_log_scale(self, key: str) -> bool:
        return key in self.hash_group_key and isinstance(
            self.hash_group_key[key].distribution,
            TransLogNormalSettings | TransLogUnifSettings,
        )

    def write_to_runpath(
        self, run_path: Path, real_nr: int, ensemble: Ensemble
    ) -> dict[str, dict[str, float]] | None:
        """
        This function is responsible for converting the parameter
        from the internal ert format to the format the forward model
        expects
        """
        df = ensemble.load_parameters_scalar(
            scalar_name=self.name, realizations=np.array([real_nr])
        ).select(pl.col("^.*\\.transformed$"))
        data_dict = df.rename(
            {col: col.replace(".transformed", "") for col in df.columns}
        ).to_dict()
        data: dict[str, dict[str, float]] = defaultdict(dict)
        for key, values in data_dict.items():
            group_name, param_name = key.split(":")
            value = values[0]
            param = self.hash_group_key[key]
            log_value: float | None = None
            if param.input_source == DataSource.SAMPLED and isinstance(
                param.distribution, TransLogNormalSettings | TransLogUnifSettings
            ):
                log_value = math.log(value, 10)

            # Build the nested dictionary {group: {key:value}, leg10_group:{key:log_value}}
            data[group_name][param_name] = value
            if log_value:
                data[f"LOG10_{group_name}"][param_name] = log_value

        outfiles: dict[str, tuple[str, str]] = {}
        for group_name, params in self.groups.items():
            for param in params:
                if param.template_file is not None and param.output_file is not None:
                    outfiles[group_name] = (param.template_file, param.output_file)
                    break
        for group_name, (template_file, output_file) in outfiles.items():
            target_file = substitute_runpath_name(
                output_file, real_nr, ensemble.iteration
            )
            if target_file.startswith("/"):
                target_file = target_file[1:]
            (run_path / target_file).parent.mkdir(exist_ok=True, parents=True)
            template_file_path = (
                ensemble.experiment.mount_point / Path(template_file).name
            )
            with open(template_file_path, encoding="utf-8") as f:
                template = f.read()
            for key, value in data[group_name].items():
                template = template.replace(f"<{key}>", f"{value:.6g}")
            with open(run_path / target_file, "w", encoding="utf-8") as f:
                f.write(template)
        return data

    @classmethod
    def from_config_list(cls, gen_kw_list: list[list[str]]) -> Self:
        errors = []
        scalars: list[ScalarParameter] = []

        for gen_kw in gen_kw_list:
            gen_kw_key = gen_kw[0]
            positional_args, options = parse_config(gen_kw, 4)
            forward_init = str_to_bool(options.get("FORWARD_INIT", "FALSE"))
            init_file = _get_abs_path(options.get("INIT_FILES"))
            update_parameter = str_to_bool(options.get("UPDATE", "TRUE"))

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
                            f"No such template file: {template_file}",
                            positional_args[1],
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
                        f"No such parameter file: {parameter_file}",
                        parameter_file_context,
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
                            scalars.append(
                                ScalarParameter(
                                    param_name=items[0],
                                    input_source=DataSource.SAMPLED,
                                    group_name=gen_kw_key,
                                    distribution=get_distribution(items[1], items[2:]),
                                    template_file=template_file,
                                    output_file=output_file,
                                    update=update_parameter,
                                )
                            )

            if gen_kw_key == "PRED" and update_parameter:
                ConfigWarning.warn(
                    "GEN_KW PRED used to hold a special meaning and be "
                    "excluded from being updated.\n If the intention was "
                    "to exclude this from updates, set UPDATE:FALSE.\n",
                    gen_kw[0],
                )
        if errors:
            raise ConfigValidationError.from_collected(errors)

        return cls(
            name=SCALAR_PARAMETERS_NAME,
            forward_init=False,
            update=False,
            scalars=scalars,
        )
