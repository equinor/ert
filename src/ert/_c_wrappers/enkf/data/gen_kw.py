import numbers
import os.path

from cwrap import BaseCClass
from ecl.util.util import DoubleVector

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.config import GenKwConfig


class GenKw(BaseCClass):
    TYPE_NAME = "gen_kw"
    _alloc = ResPrototype("void*  gen_kw_alloc(gen_kw_config)", bind=False)
    _free = ResPrototype("void   gen_kw_free(gen_kw_config)")
    _export_parameters = ResPrototype("void   gen_kw_write_export_file(gen_kw , char*)")
    _data_iget = ResPrototype("double gen_kw_data_iget(gen_kw, int, bool)")
    _data_iset = ResPrototype("void   gen_kw_data_iset(gen_kw, int, double)")
    _set_values = ResPrototype("void   gen_kw_data_set_vector(gen_kw, double_vector)")
    _data_get = ResPrototype("double gen_kw_data_get(gen_kw, char*, bool)")
    _data_set = ResPrototype("void   gen_kw_data_set(gen_kw, char*, double)")
    _size = ResPrototype("int    gen_kw_data_size(gen_kw)")
    _has_key = ResPrototype("bool   gen_kw_data_has_key(gen_kw, char*)")
    _ecl_write = ResPrototype(
        "void   gen_kw_ecl_write(gen_kw,    char* , char* , void*)"
    )
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

    def exportParameters(self, file_name):
        """@type: str"""
        self._export_parameters(file_name)

    def __getitem__(self, key):
        """
        @type key: int or str
        @rtype: float
        """
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

    def eclWrite(self, path, filename):
        if path is not None:
            if not os.path.isdir(path):
                raise IOError(f"The directory:{path} does not exist")

        self._ecl_write(path, filename, None)

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

    def __len__(self):
        """@rtype: int"""
        return self._size()

    def __contains__(self, item):
        return self._has_key(item)

    def free(self):
        self._free()

    def __repr__(self):
        return f"GenKw(len = {len(self)}) at 0x{self._address():x}"
