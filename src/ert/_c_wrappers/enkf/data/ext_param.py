from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.config import ExtParamConfig


class ExtParam(BaseCClass):
    TYPE_NAME = "ext_param"
    _alloc = ResPrototype("void*  ext_param_alloc( ext_param_config )", bind=False)
    _free = ResPrototype("void   ext_param_free( ext_param )")
    _iset = ResPrototype("void   ext_param_iset( ext_param, int, double)")
    _key_set = ResPrototype("void   ext_param_key_set( ext_param, char*, double)")
    _key_suffix_set = ResPrototype(
        "void   ext_param_key_suffix_set( ext_param, char*, char*, double)"
    )
    _iget = ResPrototype("double ext_param_iget( ext_param, int)")
    _key_get = ResPrototype("double ext_param_key_get( ext_param, char*)")
    _key_suffix_get = ResPrototype(
        "double ext_param_key_suffix_get( ext_param, char*, char*)"
    )
    _export = ResPrototype("void   ext_param_json_export( ext_param, char*)")
    _get_config = ResPrototype("void* ext_param_get_config(ext_param)")

    def __init__(self, config):
        c_ptr = self._alloc(config)
        super().__init__(c_ptr)

    def __contains__(self, key):
        return key in self.config

    def __len__(self):
        return len(self.config)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            # if the index is key suffix, assume they are both strings
            key, suffix = index
            if not isinstance(key, str) or not isinstance(suffix, str):
                raise TypeError(f"Expected a pair of strings, got {index}")
            self._check_key_suffix(key, suffix)
            return self._key_suffix_get(key, suffix)

        # index is just the key, it can be either a string or an int
        if isinstance(index, str):
            self._check_key_suffix(index)
            return self._key_get(index)

        index = self._roll_key_index(index)
        self._check_index(index)
        return self._iget(index)

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            # if the index is key suffix, assume they are both strings
            key, suffix = index
            if not isinstance(key, str) or not isinstance(suffix, str):
                raise TypeError(f"Expected a pair of strings, got {index}")
            self._check_key_suffix(key, suffix)
            self._key_suffix_set(key, suffix, value)
            return

        # index is just the key, it can be either a string or an int
        if isinstance(index, str):
            self._check_key_suffix(index)
            self._key_set(index, value)
        else:
            index = self._roll_key_index(index)
            self._check_index(index)
            self._iset(index, value)

    def _roll_key_index(self, index):
        """Support indexing from the end of the list of keys"""
        return index if index >= 0 else index + len(self)

    def _check_index(self, kidx, sidx=None):
        """Raise if any of the following is true:
        - kidx is not a valid index for keys
        - the key referred to by kidx has no suffixes, but sidx is given
        - the key referred to by kidx has suffixes, but sidx is None
        - the key referred to by kidx has suffixes, and sidx is not a valid
          suffix index
        """
        if kidx < 0 or kidx >= len(self):
            raise IndexError(
                f"Invalid key index {kidx}. Valid range is [0, {len(self)})"
            )
        key, suffixes = self.config[kidx]
        if not suffixes:
            if sidx is None:
                return  # we are good
            raise IndexError(f"Key {key} has no suffixes, but suffix {sidx} requested")
        assert len(suffixes) > 0
        if sidx is None:
            raise IndexError(
                f"Key {key} has suffixes, a suffix index must be specified"
            )
        if sidx < 0 or sidx >= len(suffixes):
            raise IndexError(
                (
                    f"Suffix index {sidx} is out of range for key {key}. "
                    f"Valid range is [0, {len(suffixes)})"
                )
            )

    def _check_key_suffix(self, key, suffix=None):
        """Raise if any of the following is true:
        - key is not present in config
        - key has no suffixes but a suffix is given
        - key has suffixes but suffix is None
        - key has suffixes but suffix is not among them
        """
        if key not in self:
            raise KeyError(f"No such key: {key}")
        suffixes = self.config[key]
        if not suffixes:
            if suffix is None:
                return
            raise KeyError(f"Key {key} has no suffixes, but suffix {suffix} requested")
        assert len(suffixes) > 0
        if suffix is None:
            raise KeyError(f"Key {key} has suffixes, a suffix must be specified")
        if suffix not in suffixes:
            raise KeyError(
                f"Key {key} has suffixes {suffixes}. "
                f"Can't find the requested suffix {suffix}"
            )

    @property
    def config(self) -> ExtParamConfig:
        return ExtParamConfig.createCReference(self._get_config(), self)

    # This could in the future be specialized to take a numpy vector,
    # which could be vector-assigned in C.
    def set_vector(self, values):
        if len(values) != len(self):
            raise ValueError("Size mismatch")

        for index, value in enumerate(values):
            self[index] = value

    def free(self):
        self._free()

    def export(self, fname):
        self._export(fname)
