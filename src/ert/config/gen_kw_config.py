from __future__ import annotations

import logging
import math
import os
import shutil
import warnings
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, TypedDict, overload

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm
from typing_extensions import Self

from ._option_dict import option_dict
from ._str_to_bool import str_to_bool
from .parameter_config import ParameterConfig
from .parsing import ConfigValidationError, ConfigWarning, ErrorInfo

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import EnsembleReader


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
class GenKwConfig(ParameterConfig):
    template_file: Optional[str]
    output_file: Optional[str]
    transfer_function_definitions: List[str]
    forward_init_file: Optional[str] = None
    template_file_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.transfer_functions: List[TransferFunction] = []
        for e in self.transfer_function_definitions:
            self.transfer_functions.append(self._parse_transfer_function(e))

        self._validate()

    def __len__(self) -> int:
        return len(self.transfer_functions)

    @classmethod
    def from_config_list(cls, gen_kw: List[str]) -> Self:
        gen_kw_key = gen_kw[0]

        if gen_kw_key == "PRED":
            ConfigWarning.ert_context_warn(
                "GEN_KW PRED used to hold a special meaning and be "
                "excluded from being updated.\n If the intention was "
                "to exclude this from updates, please use the "
                "DisableParametersUpdate workflow though the "
                "DISABLE_PARAMETERS key instead.\n",
                gen_kw[0],
            )

        options = option_dict(gen_kw, 4)
        forward_init = str_to_bool(options.get("FORWARD_INIT", "FALSE"))
        init_file = _get_abs_path(options.get("INIT_FILES"))
        errors = []

        if len(gen_kw) == 2:
            parameter_file = _get_abs_path(gen_kw[1])
            template_file = None
            output_file = None
        else:
            output_file = gen_kw[2]
            parameter_file = _get_abs_path(gen_kw[3])

            template_file = _get_abs_path(gen_kw[1])
            if not os.path.isfile(template_file):
                errors.append(
                    ConfigValidationError.with_context(
                        f"No such template file: {template_file}", gen_kw[1]
                    )
                )
        if not os.path.isfile(parameter_file):
            errors.append(
                ConfigValidationError.with_context(
                    f"No such parameter file: {parameter_file}", gen_kw[3]
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

        transfer_function_definitions: List[str] = []
        with open(parameter_file, "r", encoding="utf-8") as file:
            for item in file:
                item = item.rsplit("--")[0]  # remove comments
                if item.strip():  # only lines with content
                    transfer_function_definitions.append(item)

        return cls(
            name=gen_kw_key,
            forward_init=forward_init,
            template_file=template_file,
            output_file=output_file,
            forward_init_file=init_file,
            transfer_function_definitions=transfer_function_definitions,
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

        logging.info(f"Sampling parameter {self.name} for realization {real_nr}")
        keys = [e.name for e in self.transfer_functions]
        parameter_value = self._sample_value(
            self.name,
            keys,
            str(random_seed),
            real_nr,
            ensemble_size,
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
        keys = [e.name for e in self.transfer_functions]
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
        self, run_path: Path, real_nr: int, ensemble: EnsembleReader
    ) -> Dict[str, Dict[str, float]]:
        array = ensemble.load_parameters(self.name, real_nr, var="transformed_values")
        assert isinstance(array, xr.DataArray)
        if not array.size == len(self.transfer_functions):
            raise ValueError(
                f"The configuration of GEN_KW parameter {self.name}"
                f" is of size {len(self.transfer_functions)}, expected {array.size}"
            )

        data = dict(zip(array["names"].values.tolist(), array.values.tolist()))

        log10_data = {
            tf.name: math.log(data[tf.name], 10)
            for tf in self.transfer_functions
            if tf.use_log
        }

        if self.template_file_path is not None and self.output_file is not None:
            target_file = self.output_file
            if target_file.startswith("/"):
                target_file = target_file[1:]
            (run_path / target_file).parent.mkdir(exist_ok=True, parents=True)
            with open(self.template_file_path, "r", encoding="utf-8") as f:
                template = f.read()
            for key, value in data.items():
                template = template.replace(f"<{key}>", f"{value:.6g}")
            with open(run_path / target_file, "w", encoding="utf-8") as f:
                f.write(template)

        if log10_data:
            return {self.name: data, f"LOG10_{self.name}": log10_data}
        else:
            return {self.name: data}

    def shouldUseLogScale(self, keyword: str) -> bool:
        for tf in self.transfer_functions:
            if tf.name == keyword:
                return tf.use_log
        return False

    def getKeyWords(self) -> List[str]:
        return [tf.name for tf in self.transfer_functions]

    def get_priors(self) -> List["PriorDict"]:
        priors: List["PriorDict"] = []
        for tf in self.transfer_functions:
            priors.append(
                {
                    "key": tf.name,
                    "function": tf.transfer_function_name,
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
        for index, tf in enumerate(self.transfer_functions):
            array[index] = tf.calc_func(array[index], list(tf.parameter_list.values()))
        return array

    @staticmethod
    def _values_from_file(
        realization: int, name_format: str, keys: List[str]
    ) -> npt.NDArray[np.double]:
        file_name = name_format % realization  # noqa
        df = pd.read_csv(file_name, delim_whitespace=True, header=None)
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
        nr_samples: int,
    ) -> npt.NDArray[np.double]:
        parameter_values = []
        for key in keys:
            key_hash = sha256(
                global_seed.encode("utf-8")
                + f"{parameter_group_name}:{key}".encode("utf-8")
            )
            seed = np.frombuffer(key_hash.digest(), dtype="uint32")
            rng = np.random.default_rng(seed)
            values = rng.standard_normal(nr_samples)
            parameter_values.append(values[realization])
        return np.array(parameter_values)

    @staticmethod
    def _parse_transfer_function(param_string: str) -> TransferFunction:
        param_args = param_string.split()

        TRANS_FUNC_ARGS: dict[str, List[str]] = {
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

        if len(param_args) > 1:
            func_name = param_args[0]
            param_func_name = param_args[1]

            if (
                param_func_name not in TRANS_FUNC_ARGS
                or param_func_name not in PRIOR_FUNCTIONS
            ):
                raise ConfigValidationError(
                    f"Unknown transfer function provided: {param_func_name}"
                )

            param_names = TRANS_FUNC_ARGS[param_func_name]

            if len(param_args) - 2 != len(param_names):
                raise ConfigValidationError(
                    f"Incorrect number of values provided: {param_string}"
                )

            param_floats = []
            for p in param_args[2:]:
                try:
                    param_floats.append(float(p))
                except ValueError as e:
                    raise ConfigValidationError(
                        f"Unable to convert float number: {p}"
                    ) from e

            params = dict(zip(param_names, param_floats))

            return TransferFunction(
                name=func_name,
                transfer_function_name=param_func_name,
                parameter_list=params,
                calc_func=PRIOR_FUNCTIONS[param_func_name],
            )

        else:
            raise ConfigValidationError(
                f"Too few instructions provided in: {param_string}"
            )

    def save_experiment_data(self, experiment_path: Path) -> None:
        if self.template_file:
            incoming_template_file_path = Path(self.template_file)
            self.template_file_path = Path(
                experiment_path / incoming_template_file_path.name
            )
            shutil.copyfile(incoming_template_file_path, self.template_file_path)


@dataclass
class TransferFunction:
    name: str
    transfer_function_name: str
    parameter_list: Dict[str, float]
    calc_func: Callable[[float, List[float]], float]
    use_log: bool = False

    def __post_init__(self) -> None:
        if self.transfer_function_name in ["LOGNORMAL", "LOGUNIF"]:
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
        y = TransferFunction.trans_errf(x, [0, 1, _skew, _width])
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
    "NORMAL": TransferFunction.trans_normal,
    "LOGNORMAL": TransferFunction.trans_lognormal,
    "TRUNCATED_NORMAL": TransferFunction.trans_truncated_normal,
    "TRIANGULAR": TransferFunction.trans_triangular,
    "UNIFORM": TransferFunction.trans_unif,
    "DUNIF": TransferFunction.trans_dunif,
    "ERRF": TransferFunction.trans_errf,
    "DERRF": TransferFunction.trans_derrf,
    "LOGUNIF": TransferFunction.trans_logunif,
    "CONST": TransferFunction.trans_const,
    "RAW": TransferFunction.trans_raw,
}
