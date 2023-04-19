from __future__ import annotations

import math
import os
from hashlib import sha256
from typing import TYPE_CHECKING, Dict, Final, List, TypedDict

import numpy as np
import pandas as pd
from cwrap import BaseCClass
from ecl.util.util import StringList

from ert._c_wrappers import ResPrototype
from ert._clib import gen_kw_config
from ert.parsing.config_errors import ConfigValidationError

if TYPE_CHECKING:
    import numpy.typing as npt


class PriorDict(TypedDict):
    key: str
    function: str
    parameters: Dict[str, float]


class GenKwConfig(BaseCClass):
    TYPE_NAME = "gen_kw_config"

    _free = ResPrototype("void  gen_kw_config_free( gen_kw_config )")
    _alloc_empty = ResPrototype("void* gen_kw_config_alloc_empty()", bind=False)
    _set_parameter_file = ResPrototype(
        "void  gen_kw_config_set_parameter_file(gen_kw_config, char*)"
    )
    _alloc_name_list = ResPrototype(
        "stringlist_obj gen_kw_config_alloc_name_list(gen_kw_config)"
    )
    _should_use_log_scale = ResPrototype(
        "bool  gen_kw_config_should_use_log_scale(gen_kw_config, int)"
    )
    _size = ResPrototype("int   gen_kw_config_get_data_size(gen_kw_config)")
    _iget_name = ResPrototype("char* gen_kw_config_iget_name(gen_kw_config, int)")
    _get_function_type = ResPrototype(
        "char* gen_kw_config_iget_function_type(gen_kw_config, int)"
    )
    _get_function_parameter_names = ResPrototype(
        "stringlist_ref gen_kw_config_iget_function_parameter_names(gen_kw_config, int)"
    )

    _transform = ResPrototype(
        "double gen_kw_config_transform(gen_kw_config, int, double)"  # noqa
    )

    def __init__(
        self, key: str, template_file: str, parameter_file: str, tag_fmt: str = "<%s>"
    ):
        if not os.path.isfile(template_file):
            raise IOError(f"No such file:{template_file}")

        if not os.path.isfile(parameter_file):
            raise IOError(f"No such file:{parameter_file}")

        c_ptr = self._alloc_empty()
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError(
                "Could not instantiate GenKwConfig with "
                f'key="{key}" and tag_fmt="{tag_fmt}"'
            )

        self._key = key
        self.name = key
        self._parameter_file = parameter_file
        self._template_file = template_file
        self._tag_format = tag_fmt

        # this triggers a series of low-level events
        self._set_parameter_file(parameter_file)

        self.__str__ = self.__repr__

    def getTemplateFile(self) -> os.PathLike[str]:
        path = self._template_file
        return None if path is None else os.path.abspath(path)

    def getParameterFile(self):
        path = self._parameter_file
        return None if path is None else os.path.abspath(path)

    def getKeyWords(self) -> StringList:
        return self._alloc_name_list()

    def shouldUseLogScale(self, index: int) -> bool:
        return self._should_use_log_scale(index)

    def free(self):
        self._free()

    def __repr__(self):
        return (
            f'GenKwConfig(key = "{self.getKey()}", '
            f'tag_fmt = "{self.tag_fmt}") at 0x{self._address():x}'
        )

    def getKey(self) -> str:
        return self._key

    @property
    def tag_fmt(self):
        return self._tag_format

    def __len__(self):
        return self._size()

    def __getitem__(self, index: int) -> str:
        return self._iget_name(index)

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

    def get_priors(self) -> List["PriorDict"]:
        priors: List["PriorDict"] = []
        keys = self.getKeyWords()
        for i, key in enumerate(keys):
            function_type = self._get_function_type(i)
            parameter_names = self._get_function_parameter_names(i)
            parameter_values = gen_kw_config.get_function_parameter_values(self, i)
            priors.append(
                {
                    "key": key,
                    "function": function_type,
                    "parameters": dict(zip(parameter_names, parameter_values)),
                }
            )
        return priors

    def transform(self, index: int, value: float) -> float:
        return self._transform(index, value)

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
    def parse_transfer_function_parameters(param_string: str):
        param_args = param_string.split()

        TRANS_FUNC_ARRAY: Final = {
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

            if param_func_name not in TRANS_FUNC_ARRAY:
                raise ConfigValidationError(
                    f"Unknown transfer function provided: {param_func_name}"
                )

            param_names = TRANS_FUNC_ARRAY[param_func_name]

            if len(param_args) - 2 != len(param_names):
                raise ConfigValidationError(
                    f"Incorrect number of values provided: {param_string}"
                )

            param_floats = []
            try:
                for p in param_args[2:]:
                    param_floats.append(float(p))
            except ValueError:
                raise ConfigValidationError(f"Unable to convert float number: {p}")

            params = dict(zip(param_names, param_floats))

            tf = TransferFunction(
                func_name, param_func_name, params, TransferFunction.trans_unif
            )

            # just test
            print(tf.calculate(4))

        else:
            raise ConfigValidationError(
                f"Too few instructions provided in: {param_string}"
            )


class TransferFunction:
    name: str
    transfer_function_name: str
    param_list: List[(str, float)]
    calc_func: object

    _min: float = 0
    _max: float = 0
    _width: float = 0
    _skewness: float = 0
    _mean: float = 0
    _std: float = 0
    _xmin: float = 0
    _xmode: float = 0
    _xmax: float = 0
    _value: float = 0
    _steps: int = 0
    _use_log: bool = False

    def __init__(self, name, transfer_function_name, param_list, calc_func) -> None:
        self.name = name
        self.transfer_function_name = transfer_function_name
        self.calc_func = calc_func

        self._min = param_list.get("MIN", 0)
        self._max = param_list.get("MAX", 0)
        self._width = param_list.get("WIDTH", 0)
        self._skewness = param_list.get("SKEWNESS", 0)
        self._mean = param_list.get("MEAN", 0)
        self._std = param_list.get("STD", 0)
        self._xmin = param_list.get("XMIN", 0)
        self._xmode = param_list.get("XMODE", 0)
        self._xmax = param_list.get("XMAX", 0)
        self._value = param_list.get("VALUE", 0)
        self._steps = int(param_list.get("STEPS", 0))

        if transfer_function_name in ["LOGNORMAL", "LOGUNIF"]:
            self._use_log = True

    """
    Width  = 1 => uniform
    Width  > 1 => unimodal peaked
    Width  < 1 => bimoal peaks
    Skewness < 0 => shifts towards the left
    Skewness = 0 => symmetric
    Skewness > 0 => Shifts towards the right
    The width is a relavant scale for the value of skewness.
    """

    def trans_errf(self, x: float) -> float:
        y = 0.5 * (1 + math.erf((x + self._skewness) / (self._width * math.sqrt(2.0))))
        return self._min + y * (self._max - self._min)

    def trans_const(self, _: float) -> float:
        return self._value

    def trans_raw(self, x: float) -> float:
        return x

    '''Observe that the argument of the shift should be "+"'''

    def trans_derrf(self, x: float) -> float:
        y = math.floor(
            self._steps
            * 0.5
            * (1 + math.erf((x + self._skewness) / (self._width * math.sqrt(2.0))))
            / (self._steps - 1)
        )
        return self._min + y * (self._max - self._min)

    def trans_unif(self, x: float) -> float:
        y = 0.5 * (1 + math.erf(x / math.sqrt(2.0)))  # 0 - 1
        return y * (self._max - self._min) + self._min

    def trans_dunif(self, x: float) -> float:
        y = 0.5 * (1 + math.erf(x / math.sqrt(2.0)))  # 0 - 1
        return (math.floor(y * self._steps) / (self._steps - 1)) * (
            self._max - self._min
        ) + self._min

    def trans_normal(self, x: float) -> float:
        return x * self._std + self._mean

    def trans_truncated_normal(self, x: float) -> float:
        y = x * self._std + self._mean
        max(min(y, self._max), self._min)  # clamp
        return y

    def trans_lognormal(self, x: float) -> float:
        # mean is the expectation of log( y )
        return math.exp(x * self._std + self._mean)

    def trans_logunif(self, x: float) -> float:
        tmp = 0.5 * (1 + math.erf(x / math.sqrt(2.0)))  # 0 - 1
        log_y = self._min + tmp * (
            self._max - self._min
        )  # Shift according to max / min
        return math.exp(log_y)

    def trans_triangular(self, x: float) -> float:
        inv_norm_left = (self._xmax - self._xmin) * (self._xmode - self._xmin)
        inv_norm_right = (self._xmax - self._xmin) * (self._xmax - self._xmode)
        ymode = (self._xmode - self._xmin) / (self._xmax - self._xmin)
        y = 0.5 * (1 + math.erf(x / math.sqrt(2.0)))  # 0 - 1

        if y < ymode:
            return self._xmin + math.sqrt(y * inv_norm_left)
        else:
            return self._xmax - math.sqrt((1 - y) * inv_norm_right)

    def calculate(self, x) -> float:
        return self.calc_func(self, x)
