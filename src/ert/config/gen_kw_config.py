from __future__ import annotations

import logging
import math
import os
import shutil
import warnings
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypedDict,
    overload,
)

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm
from typing_extensions import Self

from ._str_to_bool import str_to_bool
from .parameter_config import ParameterConfig, parse_config
from .parsing import ConfigValidationError, ConfigWarning, ErrorInfo

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import Ensemble

_logger = logging.getLogger(__name__)


class PriorDict(TypedDict):
    key: str
    function: str
    parameters: Dict[str, float]


@overload
def _get_abs_path(file: None) -> None:
    pass


@overload
def _get_abs_path(file: str) -> str:
    pass


def _get_abs_path(file: Optional[str]) -> Optional[str]:
    if file is not None:
        file = os.path.realpath(file)
    return file


@dataclass
class TransformFunctionDefinition:
    name: str
    param_name: str
    values: List[Any]


@dataclass
class GenKwConfig(ParameterConfig):
    template_file: Optional[str]
    output_file: Optional[str]
    transform_function_definitions: (
        List[TransformFunctionDefinition] | List[Dict[Any, Any]]
    )
    forward_init_file: Optional[str] = None

    def __post_init__(self) -> None:
        self.transform_functions: List[TransformFunction] = []
        for e in self.transform_function_definitions:
            if isinstance(e, dict):
                self.transform_functions.append(
                    self._parse_transform_function_definition(
                        TransformFunctionDefinition(**e)
                    )
                )
            else:
                self.transform_functions.append(
                    self._parse_transform_function_definition(e)
                )
        self._validate()

    def __len__(self) -> int:
        return len(self.transform_functions)

    @classmethod
    def from_config_list(cls, gen_kw: List[str]) -> Self:
        gen_kw_key = gen_kw[0]

        positional_args, options = parse_config(gen_kw, 4)
        forward_init = str_to_bool(options.get("FORWARD_INIT", "FALSE"))
        init_file = _get_abs_path(options.get("INIT_FILES"))
        update_parameter = str_to_bool(options.get("UPDATE", "TRUE"))
        errors = []

        if len(positional_args) == 2:
            parameter_file = _get_abs_path(positional_args[1])
            template_file = None
            output_file = None
        elif len(positional_args) == 4:
            output_file = positional_args[2]
            parameter_file = _get_abs_path(positional_args[3])

            template_file = _get_abs_path(positional_args[1])
            if not os.path.isfile(template_file):
                errors.append(
                    ConfigValidationError.with_context(
                        f"No such template file: {template_file}", positional_args[1]
                    )
                )
        else:
            raise ConfigValidationError(
                f"Unexpected positional arguments: {positional_args}"
            )
        if not os.path.isfile(parameter_file):
            errors.append(
                ConfigValidationError.with_context(
                    f"No such parameter file: {parameter_file}", positional_args[3]
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

        if init_file and "%" not in init_file:
            errors.append(
                ConfigValidationError.with_context(
                    "Loading GEN_KW from files requires %d in file format", gen_kw
                )
            )
        if errors:
            raise ConfigValidationError.from_collected(errors)

        transform_function_definitions: List[TransformFunctionDefinition] = []
        with open(parameter_file, "r", encoding="utf-8") as file:
            for item in file:
                item = item.split("--")[0]  # remove comments
                if item.strip():  # only lines with content
                    items = item.split()
                    transform_function_definitions.append(
                        TransformFunctionDefinition(
                            name=items[0],
                            param_name=items[1],
                            values=items[2:],
                        )
                    )

        if gen_kw_key == "PRED" and update_parameter:
            ConfigWarning.ert_context_warn(
                "GEN_KW PRED used to hold a special meaning and be "
                "excluded from being updated.\n If the intention was "
                "to exclude this from updates, set UPDATE:FALSE.\n",
                gen_kw[0],
            )
        return cls(
            name=gen_kw_key,
            forward_init=forward_init,
            template_file=template_file,
            output_file=output_file,
            forward_init_file=init_file,
            transform_function_definitions=transform_function_definitions,
            update=update_parameter,
        )

    def _validate(self) -> None:
        errors = []

        def _check_non_negative_parameter(param: str, prior: PriorDict) -> None:
            key = prior["key"]
            dist = prior["function"]
            param_val = prior["parameters"][param]
            if param_val < 0:
                errors.append(
                    ErrorInfo(
                        f"Negative {param} {param_val!r}"
                        f" for {dist} distributed parameter {key!r}",
                    ).set_context(self.name)
                )

        unique_keys = set()
        for prior in self.get_priors():
            key = prior["key"]
            if key in unique_keys:
                errors.append(
                    ErrorInfo(
                        f"Duplicate GEN_KW keys {key!r} found, keys must be unique."
                    ).set_context(self.name)
                )
            unique_keys.add(key)

            if prior["function"] == "LOGNORMAL":
                _check_non_negative_parameter("MEAN", prior)
                _check_non_negative_parameter("STD", prior)
            elif prior["function"] in ["NORMAL", "TRUNCATED_NORMAL"]:
                _check_non_negative_parameter("STD", prior)
        if errors:
            raise ConfigValidationError.from_collected(errors)

    def sample_or_load(
        self, real_nr: int, random_seed: int, ensemble_size: int
    ) -> xr.Dataset:
        if self.forward_init_file:
            return self.read_from_runpath(Path(), real_nr)

        _logger.info(f"Sampling parameter {self.name} for realization {real_nr}")
        keys = [e.name for e in self.transform_functions]
        parameter_value = self._sample_value(
            self.name,
            keys,
            str(random_seed),
            real_nr,
        )

        return xr.Dataset(
            {
                "values": ("names", parameter_value),
                "transformed_values": ("names", self.transform(parameter_value)),
                "names": keys,
            }
        )

    def read_from_runpath(
        self,
        run_path: Path,
        real_nr: int,
    ) -> xr.Dataset:
        keys = [e.name for e in self.transform_functions]
        if not self.forward_init_file:
            raise ValueError("loading gen_kw values requires forward_init_file")

        parameter_value = self._values_from_file(
            real_nr,
            self.forward_init_file,
            keys,
        )

        return xr.Dataset(
            {
                "values": ("names", parameter_value),
                "transformed_values": ("names", self.transform(parameter_value)),
                "names": keys,
            }
        )

    def write_to_runpath(
        self,
        run_path: Path,
        real_nr: int,
        ensemble: Ensemble,
    ) -> Dict[str, Dict[str, float]]:
        array = ensemble.load_parameters(self.name, real_nr)["transformed_values"]
        assert isinstance(array, xr.DataArray)
        if not array.size == len(self.transform_functions):
            raise ValueError(
                f"The configuration of GEN_KW parameter {self.name}"
                f" is of size {len(self.transform_functions)}, expected {array.size}"
            )

        data = dict(zip(array["names"].values.tolist(), array.values.tolist()))

        log10_data = {
            tf.name: math.log(data[tf.name], 10)
            for tf in self.transform_functions
            if tf.use_log
        }

        if self.template_file is not None and self.output_file is not None:
            target_file = self.output_file
            if target_file.startswith("/"):
                target_file = target_file[1:]
            (run_path / target_file).parent.mkdir(exist_ok=True, parents=True)
            template_file_path = (
                ensemble.experiment.mount_point / Path(self.template_file).name
            )
            with open(template_file_path, "r", encoding="utf-8") as f:
                template = f.read()
            for key, value in data.items():
                template = template.replace(f"<{key}>", f"{value:.6g}")
            with open(run_path / target_file, "w", encoding="utf-8") as f:
                f.write(template)

        if log10_data:
            return {self.name: data, f"LOG10_{self.name}": log10_data}
        else:
            return {self.name: data}

    def save_parameters(
        self,
        ensemble: Ensemble,
        group: str,
        realization: int,
        data: npt.NDArray[np.float64],
    ) -> None:
        ds = xr.Dataset(
            {
                "values": ("names", data),
                "transformed_values": (
                    "names",
                    self.transform(data),
                ),
                "names": [e.name for e in self.transform_functions],
            }
        )
        ensemble.save_parameters(group, realization, ds)

    @staticmethod
    def load_parameters(
        ensemble: Ensemble, group: str, realizations: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        return ensemble.load_parameters(group, realizations)["values"].values.T

    def shouldUseLogScale(self, keyword: str) -> bool:
        for tf in self.transform_functions:
            if tf.name == keyword:
                return tf.use_log
        return False

    def getKeyWords(self) -> List[str]:
        return [tf.name for tf in self.transform_functions]

    def get_priors(self) -> List["PriorDict"]:
        priors: List["PriorDict"] = []
        for tf in self.transform_functions:
            priors.append(
                {
                    "key": tf.name,
                    "function": tf.transform_function_name,
                    "parameters": tf.parameter_list,
                }
            )

        return priors

    def transform(self, array: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Transform the input array in accordance with priors

        Parameters:
            array: An array of standard normal values

        Returns: Transformed array, where each element has been transformed from
            a standard normal distribution to the distribution set by the user
        """
        array = np.array(array)
        for index, tf in enumerate(self.transform_functions):
            array[index] = tf.calc_func(array[index], list(tf.parameter_list.values()))
        return array

    @staticmethod
    def _values_from_file(
        realization: int, name_format: str, keys: List[str]
    ) -> npt.NDArray[np.double]:
        file_name = name_format % realization  # noqa
        df = pd.read_csv(file_name, sep=r"\s+", header=None)
        # This means we have a key: value mapping in the
        # file otherwise it is just a list of values
        if df.shape[1] == 2:
            # We need to sort the user input keys by the
            # internal order of sub-parameters:
            df = df.set_index(df.columns[0])
            return df.reindex(keys).values.flatten()
        if not np.issubdtype(df.values.dtype, np.number):
            raise ValueError(
                f"The file {file_name} did not contain numbers, got {df.values.dtype}"
            )
        return df.values.flatten()

    @staticmethod
    def _sample_value(
        parameter_group_name: str,
        keys: List[str],
        global_seed: str,
        realization: int,
    ) -> npt.NDArray[np.double]:
        """
        Generate a sample value for each key in a parameter group.

        The sampling is reproducible and dependent on a global seed combined
        with the parameter group name and individual key names. The 'realization' parameter
        determines the specific sample point from the distribution for each parameter.

        Parameters:
        - parameter_group_name (str): The name of the parameter group, used to ensure unique RNG
        seeds for different groups.
        - keys (List[str]): A list of parameter keys for which the sample values are generated.
        - global_seed (str): A global seed string used for RNG seed generation to ensure
        reproducibility across runs.
        - realization (int): An integer used to advance the RNG to a specific point in its
        sequence, effectively selecting the 'realization'-th sample from the distribution.

        Returns:
        - npt.NDArray[np.double]: An array of sample values, one for each key in the provided list.

        Note:
        The method uses SHA-256 for hash generation and numpy's default random number generator
        for sampling. The RNG state is advanced to the 'realization' point before generating
        a single sample, enhancing efficiency by avoiding the generation of large, unused sample sets.
        """
        parameter_values = []
        for key in keys:
            key_hash = sha256(
                global_seed.encode("utf-8")
                + f"{parameter_group_name}:{key}".encode("utf-8")
            )
            seed = np.frombuffer(key_hash.digest(), dtype="uint32")
            rng = np.random.default_rng(seed)

            # Advance the RNG state to the realization point
            rng.standard_normal(realization)

            # Generate a single sample
            value = rng.standard_normal(1)
            parameter_values.append(value[0])
        return np.array(parameter_values)

    @staticmethod
    def _parse_transform_function_definition(
        t: TransformFunctionDefinition,
    ) -> TransformFunction:
        if t.param_name is None and t.values is None:
            raise ConfigValidationError(f"Too few instructions provided in: {t}")

        if (
            t.param_name not in DISTRIBUTION_PARAMETERS
            or t.param_name not in PRIOR_FUNCTIONS
        ):
            raise ConfigValidationError(
                f"Unknown transform function provided: {t.param_name}"
            )

        param_names = DISTRIBUTION_PARAMETERS[t.param_name]

        if len(t.values) != len(param_names):
            raise ConfigValidationError(
                f"Incorrect number of values provided: {t.values} "
            )

        param_floats = []
        for p in t.values:
            try:
                param_floats.append(float(p))
            except ValueError as e:
                raise ConfigValidationError(
                    f"Unable to convert float number: {p}"
                ) from e

        params = dict(zip(param_names, param_floats))

        return TransformFunction(
            name=t.name,
            transform_function_name=t.param_name,
            parameter_list=params,
            calc_func=PRIOR_FUNCTIONS[t.param_name],
        )

    def save_experiment_data(self, experiment_path: Path) -> None:
        if self.template_file:
            incoming_template_file_path = Path(self.template_file)
            template_file_path = Path(
                experiment_path / incoming_template_file_path.name
            )
            shutil.copyfile(incoming_template_file_path, template_file_path)


@dataclass
class TransformFunction:
    name: str
    transform_function_name: str
    parameter_list: Dict[str, float]
    calc_func: Callable[[float, List[float]], float]
    use_log: bool = False

    def __post_init__(self) -> None:
        if self.transform_function_name in ["LOGNORMAL", "LOGUNIF"]:
            self.use_log = True

    @staticmethod
    def trans_errf(x: float, arg: List[float]) -> float:
        """
        Width  = 1 => uniform
        Width  > 1 => unimodal peaked
        Width  < 1 => bimodal peaks
        Skewness < 0 => shifts towards the left
        Skewness = 0 => symmetric
        Skewness > 0 => Shifts towards the right
        The width is a relavant scale for the value of skewness.
        """
        _min, _max, _skew, _width = arg[0], arg[1], arg[2], arg[3]
        y = norm(loc=0, scale=_width).cdf(x + _skew)
        if np.isnan(y):
            raise ValueError(
                (
                    "Output is nan, likely from triplet (x, skewness, width) "
                    "leading to low/high-probability in normal CDF."
                )
            )
        return _min + y * (_max - _min)

    @staticmethod
    def trans_const(_: float, arg: List[float]) -> float:
        return arg[0]

    @staticmethod
    def trans_raw(x: float, _: List[float]) -> float:
        return x

    @staticmethod
    def trans_derrf(x: float, arg: List[float]) -> float:
        """
        Bin the result of `trans_errf` with `min=0` and `max=1` to closest of `nbins`
        linearly spaced values on [0,1]. Finally map [0,1] to [min, max].
        """
        _steps, _min, _max, _skew, _width = (
            int(arg[0]),
            arg[1],
            arg[2],
            arg[3],
            arg[4],
        )
        q_values = np.linspace(start=0, stop=1, num=_steps)
        q_checks = np.linspace(start=0, stop=1, num=_steps + 1)[1:]
        y = TransformFunction.trans_errf(x, [0, 1, _skew, _width])
        bin_index = np.digitize(y, q_checks, right=True)
        y_binned = q_values[bin_index]
        result = _min + y_binned * (_max - _min)
        if result > _max or result < _min:
            warnings.warn(
                "trans_derff suffered from catastrophic loss of precision, clamping to min,max",
                stacklevel=1,
            )
            return np.clip(result, _min, _max)
        if np.isnan(result):
            raise ValueError(
                "trans_derrf returns nan, check that input arguments are reasonable"
            )
        return result

    @staticmethod
    def trans_unif(x: float, arg: List[float]) -> float:
        _min, _max = arg[0], arg[1]
        y = norm.cdf(x)
        return y * (_max - _min) + _min

    @staticmethod
    def trans_dunif(x: float, arg: List[float]) -> float:
        _steps, _min, _max = int(arg[0]), arg[1], arg[2]
        y = norm.cdf(x)
        return (math.floor(y * _steps) / (_steps - 1)) * (_max - _min) + _min

    @staticmethod
    def trans_normal(x: float, arg: List[float]) -> float:
        _mean, _std = arg[0], arg[1]
        return x * _std + _mean

    @staticmethod
    def trans_truncated_normal(x: float, arg: List[float]) -> float:
        _mean, _std, _min, _max = arg[0], arg[1], arg[2], arg[3]
        y = x * _std + _mean
        return max(min(y, _max), _min)  # clamp

    @staticmethod
    def trans_lognormal(x: float, arg: List[float]) -> float:
        # mean is the expectation of log( y )
        _mean, _std = arg[0], arg[1]
        return math.exp(x * _std + _mean)

    @staticmethod
    def trans_logunif(x: float, arg: List[float]) -> float:
        _log_min, _log_max = math.log(arg[0]), math.log(arg[1])
        tmp = norm.cdf(x)
        log_y = _log_min + tmp * (_log_max - _log_min)  # Shift according to max / min
        return math.exp(log_y)

    @staticmethod
    def trans_triangular(x: float, arg: List[float]) -> float:
        _min, _mode, _max = arg[0], arg[1], arg[2]
        inv_norm_left = (_max - _min) * (_mode - _min)
        inv_norm_right = (_max - _min) * (_max - _mode)
        ymode = (_mode - _min) / (_max - _min)
        y = norm.cdf(x)

        if y < ymode:
            return _min + math.sqrt(y * inv_norm_left)
        else:
            return _max - math.sqrt((1 - y) * inv_norm_right)

    def calculate(self, x: float, arg: List[float]) -> float:
        return self.calc_func(x, arg)


PRIOR_FUNCTIONS: dict[str, Callable[[float, List[float]], float]] = {
    "NORMAL": TransformFunction.trans_normal,
    "LOGNORMAL": TransformFunction.trans_lognormal,
    "TRUNCATED_NORMAL": TransformFunction.trans_truncated_normal,
    "TRIANGULAR": TransformFunction.trans_triangular,
    "UNIFORM": TransformFunction.trans_unif,
    "DUNIF": TransformFunction.trans_dunif,
    "ERRF": TransformFunction.trans_errf,
    "DERRF": TransformFunction.trans_derrf,
    "LOGUNIF": TransformFunction.trans_logunif,
    "CONST": TransformFunction.trans_const,
    "RAW": TransformFunction.trans_raw,
}


DISTRIBUTION_PARAMETERS: dict[str, List[str]] = {
    "NORMAL": ["MEAN", "STD"],
    "LOGNORMAL": ["MEAN", "STD"],
    "TRUNCATED_NORMAL": ["MEAN", "STD", "MIN", "MAX"],
    "TRIANGULAR": ["MIN", "MODE", "MAX"],
    "UNIFORM": ["MIN", "MAX"],
    "DUNIF": ["STEPS", "MIN", "MAX"],
    "ERRF": ["MIN", "MAX", "SKEWNESS", "WIDTH"],
    "DERRF": ["STEPS", "MIN", "MAX", "SKEWNESS", "WIDTH"],
    "LOGUNIF": ["MIN", "MAX"],
    "CONST": ["VALUE"],
    "RAW": [],
}
