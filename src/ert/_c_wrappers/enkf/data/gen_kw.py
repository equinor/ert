import numbers
from hashlib import sha256
from typing import List, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from cwrap import BaseCClass
from ecl.util.util import DoubleVector

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.config import GenKwConfig


class GenKw(BaseCClass):
    TYPE_NAME = "gen_kw"
    _alloc = ResPrototype("void*  gen_kw_alloc(gen_kw_config)", bind=False)
    _free = ResPrototype("void   gen_kw_free(gen_kw_config)")
    _data_iget = ResPrototype("double gen_kw_data_iget(gen_kw, int, bool)")
    _data_iset = ResPrototype("void   gen_kw_data_iset(gen_kw, int, double)")
    _set_values = ResPrototype("void   gen_kw_data_set_vector(gen_kw, double_vector)")
    _data_get = ResPrototype("double gen_kw_data_get(gen_kw, char*, bool)")
    _data_set = ResPrototype("void   gen_kw_data_set(gen_kw, char*, double)")
    _size = ResPrototype("int    gen_kw_data_size(gen_kw)")
    _has_key = ResPrototype("bool   gen_kw_data_has_key(gen_kw, char*)")
    _iget_key = ResPrototype("char*  gen_kw_get_name(gen_kw, int)")

    def __init__(self, gen_kw_config: GenKwConfig):
        """
        @type gen_kw_config: GenKwConfig
        """
        c_ptr = self._alloc(gen_kw_config)

        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError(
                f"Cannot issue a GenKw from the given keyword config: {gen_kw_config}."
            )

    def __str__(self):
        return repr(self)

    def __getitem__(self, key: Union[str, int]) -> float:
        do_transform = False
        if isinstance(key, str):
            if key not in self:
                raise KeyError(f"Key {key} does not exist")
            return self._data_get(key, do_transform)
        elif isinstance(key, int):
            if not 0 <= key < len(self):
                raise IndexError(f"Index out of range 0 <= {key} < {len(self)}")
            return self._data_iget(key, do_transform)
        else:
            raise TypeError(
                f"Illegal type for indexing, must be int or str, got: {key}"
            )

    def __setitem__(self, key, value):
        """
        @type key: int or str
        @type value: float
        """
        if isinstance(key, str):
            if key not in self:
                raise KeyError(f"Key {key} does not exist")
            self._data_set(key, value)
        elif isinstance(key, int):
            if not 0 <= key < len(self):
                raise IndexError(f"Index out of range 0 <= {key} < {len(self)}")
            self._data_iset(key, value)
        else:
            raise TypeError(
                f"Illegal type for indexing, must be int or str, got: {key}"
            )

    def items(self):
        do_transform = False
        v = []
        for index in range(len(self)):
            v.append((self._iget_key(index), self._data_iget(index, do_transform)))
        return v

    def setValues(self, values):
        if len(values) == len(self):
            if isinstance(values, DoubleVector):
                self._set_values(values)
            else:
                d = DoubleVector()
                for (index, v) in enumerate(values):
                    if isinstance(v, numbers.Number):
                        d[index] = v
                    else:
                        raise TypeError(f"Values must numeric: {v} is invalid")
                self._set_values(d)
        else:
            raise ValueError("Size mismatch between GenKW and values")

    def __len__(self) -> int:
        return self._size()

    def __contains__(self, item):
        return self._has_key(item)

    def free(self):
        self._free()

    def __repr__(self):
        return f"GenKw(len = {len(self)}) at 0x{self._address():x}"

    @staticmethod
    def values_from_files(
        realizations: List[int], name_format: str, keys: List[str]
    ) -> npt.ArrayLike:
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
    ) -> npt.ArrayLike:
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
