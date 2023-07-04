from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
import xarray as xr
from jinja2 import Template

from ert._c_wrappers.enkf.config.parameter_config import ParameterConfig
from ert.parsing.config_errors import ConfigValidationError

if TYPE_CHECKING:
    import numpy.typing as npt
    from numpy.random import SeedSequence

    from ert.storage import EnsembleReader


class PriorDict(TypedDict):
    key: str
    function: str
    parameters: Dict[str, float]


@dataclass
class GenKwConfig(ParameterConfig):
    template_file: str
    parameter_file: str
    output_file: str
    transfer_function_definitions: List[str]
    forward_init_file: Optional[str] = None

    def __post_init__(self):
        self.transfer_functions: List[TransferFunction] = []
        for e in self.transfer_function_definitions:
            self.transfer_functions.append(self._parse_transfer_function(e))

    def sample_or_load(
        self, real_nr: int, random_seed: SeedSequence, ensemble_size: int
    ):
        if self.forward_init_file:
            return self.read_from_runpath(Path(), real_nr)

        logging.info(f"Sampling parameter {self.name} for realization {real_nr}")
        keys = [e.name for e in self.transfer_functions]
        parameter_value = self._sample_value(
            self.name,
            keys,
            str(random_seed.entropy),
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
        if not array.size == len(self.transfer_functions):
            raise ValueError(
                f"The configuration of GEN_KW parameter {self.name}"
                f" is of size {len(self.transfer_functions)}, expected {array.size}"
            )

        with open(self.template_file, "r", encoding="utf-8") as f:
            template = Template(
                f.read(), variable_start_string="<", variable_end_string=">"
            )
        data = dict(zip(array["names"].values.tolist(), array.values.tolist()))
        log10_data = {
            tf.name: math.log(data[tf.name], 10)
            for tf in self.transfer_functions
            if tf.use_log
        }

        target_file = self.output_file
        if self.output_file.startswith("/"):
            target_file = self.output_file[1:]

        (run_path / target_file).parent.mkdir(exist_ok=True, parents=True)
        with open(run_path / target_file, "w", encoding="utf-8") as f:
            f.write(
                template.render({key: f"{value:.6g}" for key, value in data.items()})
            )
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

    def transform(self, array: npt.ArrayLike[np.float64]) -> npt.NDArray[np.float64]:
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
        file_name = name_format % realization
        df = pd.read_csv(file_name, delim_whitespace=True, header=None)
        # This means we have a key: value mapping in the
        # file otherwise it is just a list of values
        if df.shape[1] == 2:
            # We need to sort the user input keys by the
            # internal order of sub-parameters:
            df = df.set_index(df.columns[0])
            return df.reindex(keys).values.flatten()
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
            "TRIANGULAR": ["XMIN", "XMODE", "XMAX"],
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
                except ValueError:
                    raise ConfigValidationError(f"Unable to convert float number: {p}")

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
    def trans_errf(x, arg: List[float]) -> float:
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
        y = 0.5 * (1 + math.erf((x + _skew) / (_width * math.sqrt(2.0))))
        return _min + y * (_max - _min)

    @staticmethod
    def trans_const(_: float, arg: List[float]) -> float:
        return arg[0]

    @staticmethod
    def trans_raw(x: float, _: List[float]) -> float:
        return x

    @staticmethod
    def trans_derrf(x: float, arg: List[float]) -> float:
        '''Observe that the argument of the shift should be \"+\"'''
        _steps, _min, _max, _skew, _width = int(arg[0]), arg[1], arg[2], arg[3], arg[4]
        y = math.floor(
            _steps
            * 0.5
            * (1 + math.erf((x + _skew) / (_width * math.sqrt(2.0))))
            / (_steps - 1)
        )
        return _min + y * (_max - _min)

    @staticmethod
    def trans_unif(x: float, arg: List[float]) -> float:
        _min, _max = arg[0], arg[1]
        y = 0.5 * (1 + math.erf(x / math.sqrt(2.0)))  # 0 - 1
        return y * (_max - _min) + _min

    @staticmethod
    def trans_dunif(x: float, arg: List[float]) -> float:
        _steps, _min, _max = int(arg[0]), arg[1], arg[2]
        y = 0.5 * (1 + math.erf(x / math.sqrt(2.0)))  # 0 - 1
        return (math.floor(y * _steps) / (_steps - 1)) * (_max - _min) + _min

    @staticmethod
    def trans_normal(x: float, arg: List[float]) -> float:
        _mean, _std = arg[0], arg[1]
        return x * _std + _mean

    @staticmethod
    def trans_truncated_normal(x: float, arg: List[float]) -> float:
        _mean, _std, _min, _max = arg[0], arg[1], arg[2], arg[3]
        y = x * _std + _mean
        max(min(y, _max), _min)  # clamp
        return y

    @staticmethod
    def trans_lognormal(x: float, arg: List[float]) -> float:
        # mean is the expectation of log( y )
        _mean, _std = arg[0], arg[1]
        return math.exp(x * _std + _mean)

    @staticmethod
    def trans_logunif(x: float, arg: List[float]) -> float:
        _log_min, _log_max = math.log(arg[0]), math.log(arg[1])
        tmp = 0.5 * (1 + math.erf(x / math.sqrt(2.0)))  # 0 - 1
        log_y = _log_min + tmp * (_log_max - _log_min)  # Shift according to max / min
        return math.exp(log_y)

    @staticmethod
    def trans_triangular(x: float, arg: List[float]) -> float:
        _xmin, _xmode, _xmax = arg[0], arg[1], arg[2]
        inv_norm_left = (_xmax - _xmin) * (_xmode - _xmin)
        inv_norm_right = (_xmax - _xmin) * (_xmax - _xmode)
        ymode = (_xmode - _xmin) / (_xmax - _xmin)
        y = 0.5 * (1 + math.erf(x / math.sqrt(2.0)))  # 0 - 1

        if y < ymode:
            return _xmin + math.sqrt(y * inv_norm_left)
        else:
            return _xmax - math.sqrt((1 - y) * inv_norm_right)

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
