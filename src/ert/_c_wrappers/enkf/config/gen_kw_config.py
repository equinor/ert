from __future__ import annotations

import math
import os
from hashlib import sha256
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, TypedDict

import numpy as np
import pandas as pd

from ert.parsing.config_errors import ConfigValidationError

if TYPE_CHECKING:
    import numpy.typing as npt


class PriorDict(TypedDict):
    key: str
    function: str
    parameters: Dict[str, float]


class GenKwConfig:
    TYPE_NAME = "gen_kw_config"

    def __init__(
        self, key: str, template_file: str, parameter_file: str, tag_fmt: str = "<%s>"
    ):
        if not os.path.isfile(template_file):
            raise IOError(f"No such file:{template_file}")

        if not os.path.isfile(parameter_file):
            raise IOError(f"No such file:{parameter_file}")

        self.name = key
        self._parameter_file = parameter_file
        self._template_file = template_file
        self._tag_format = tag_fmt
        self._transfer_functions = []
        self.forward_init = ""
        self.output_file = ""
        self.forward_init_file = ""

        with open(parameter_file, "r", encoding="utf-8") as file:
            for item in file:
                item = item.rsplit("--")[0]  # remove comments

                if item.strip():  # only lines with content
                    self._transfer_functions.append(self.parse_transfer_function(item))

    def getTemplateFile(self) -> str:
        path = self._template_file
        return None if path is None else os.path.abspath(path)

    def getParameterFile(self) -> str:
        path = self._parameter_file
        return None if path is None else os.path.abspath(path)

    def shouldUseLogScale(self, keyword: str) -> bool:
        for tf in self._transfer_functions:
            if tf.name == keyword:
                return tf._use_log
        return False

    def __repr__(self):
        return f'GenKwConfig(key = "{self.getKey()}", ' f'tag_fmt = "{self.tag_fmt}")'

    def getKey(self) -> str:
        return self.name

    def getImplementationType(self):
        return self

    @property
    def tag_fmt(self):
        return self._tag_format

    def __len__(self):
        return len(self._transfer_functions)

    def __getitem__(self, index: int) -> str:
        return self._transfer_functions[index].name

    def __iter__(self):
        index = 0
        while index < len(self):
            yield self[index]
            index += 1

    def __eq__(self, other) -> bool:
        if self.getTemplateFile() != other.getTemplateFile():
            return False

        if self.getParameterFile() != other.getParameterFile():
            return False

        if self.getKey() != other.getKey():
            return False

        return True

    def getKeyWords(self) -> List[str]:
        return [tf.name for tf in self._transfer_functions]

    def get_priors(self) -> List["PriorDict"]:
        priors: List["PriorDict"] = []
        for tf in self._transfer_functions:
            priors.append(
                {
                    "key": tf.name,
                    "function": tf.transfer_function_name,
                    "parameters": tf.parameter_list,
                }
            )

        return priors

    @staticmethod
    def values_from_files(
        realizations: List[int], name_format: str, keys: List[str]
    ) -> npt.NDArray[np.double]:
        df_values = pd.DataFrame()
        for iens in realizations:
            df = pd.read_csv(
                name_format % iens,
                delim_whitespace=True,
                header=None,
            )
            # This means we have a key: value mapping in the
            # file otherwise it is just a list of values
            if df.shape[1] == 2:
                # We need to sort the user input keys by the
                # internal order of sub-parameters:
                df = df.set_index(df.columns[0])
                values = df.reindex(keys).values.flatten()
            else:
                values = df.values.flatten()
            df_values[f"{iens}"] = values
        return df_values.values

    @staticmethod
    def sample_values(
        parameter_group_name: str,
        keys: List[str],
        global_seed: str,
        active_realizations: List[int],
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
            if len(active_realizations) != nr_samples:
                values = values[active_realizations]
            parameter_values.append(values)
        return np.array(parameter_values)

    @staticmethod
    def parse_transfer_function(param_string: str) -> TransferFunction:
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
                func_name, param_func_name, params, PRIOR_FUNCTIONS[param_func_name]
            )

        else:
            raise ConfigValidationError(
                f"Too few instructions provided in: {param_string}"
            )


class TransferFunction:
    name: str
    transfer_function_name: str
    param_list: List[Tuple[str, float]]
    calc_func: Callable[[float, List[float]], float]
    _use_log: bool = False

    def __init__(self, name, transfer_function_name, param_list, calc_func) -> None:
        self.name = name
        self.transfer_function_name = transfer_function_name
        self.calc_func = calc_func
        self.parameter_list = param_list

        if transfer_function_name in ["LOGNORMAL", "LOGUNIF"]:
            self._use_log = True

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
