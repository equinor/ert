from cwrap import BaseCClass
from ecl.util.util import StringList

from ert._c_wrappers import ResPrototype


class ExtParamConfig(BaseCClass):
    TYPE_NAME = "ext_param_config"
    _alloc = ResPrototype(
        "void*   ext_param_config_alloc( char*, stringlist )", bind=False
    )
    _size = ResPrototype("int     ext_param_config_get_data_size( ext_param_config )")
    _iget_key = ResPrototype(
        "char*   ext_param_config_iget_key( ext_param_config , int)"
    )
    _free = ResPrototype("void    ext_param_config_free( ext_param_config )")
    _has_key = ResPrototype(
        "bool    ext_param_config_has_key( ext_param_config , char* )"
    )
    _key_index = ResPrototype(
        "int     ext_param_config_get_key_index(ext_param_config, char*)"
    )
    _ikey_get_suffix_count = ResPrototype(
        "int   ext_param_config_ikey_get_suffix_count(ext_param_config, int)"
    )
    _ikey_iget_suffix = ResPrototype(
        "char* ext_param_config_ikey_iget_suffix(ext_param_config, int, int)"
    )
    _ikey_set_suffixes = ResPrototype(
        "void ext_param_config_ikey_set_suffixes(ext_param_config, int, stringlist)"
    )

    def __init__(self, key, input_keys):
        """Create an ExtParamConfig for @key with the given @input_keys

        @input_keys can be either a list of keys as strings or a dict with
        keys as strings and a list of suffixes for each key.
        If a list of strings is given, the order is preserved.
        """
        try:
            keys = input_keys.keys()  # extract keys if suffixes are also given
            suffixmap = input_keys.items()
        except AttributeError:
            keys = input_keys  # assume list of keys
            suffixmap = {}

        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate keys for key '{key}' - keys: {keys}")

        keys = StringList(initial=input_keys)
        c_ptr = self._alloc(key, keys)
        super().__init__(c_ptr)

        for k, suffixes in suffixmap:
            suffixlist = StringList(initial=suffixes)
            if len(suffixes) == 0:
                raise ValueError(
                    f"No suffixes for key '{key}/{k}' - suffixes: {suffixes}"
                )
            if len(suffixes) != len(set(suffixes)):
                raise ValueError(
                    f"Duplicate suffixes for key '{key}/{k}' - " f"suffixes: {suffixes}"
                )
            if any(len(s) == 0 for s in suffixes):
                raise ValueError(
                    f"Empty suffix encountered for key '{key}/{k}' "
                    f"- suffixes: {suffixes}"
                )
            self._ikey_set_suffixes(self._key_index(k), suffixlist)

    def __len__(self):
        return self._size()

    def __contains__(self, key):
        """Check if the @key is present in the configuration

        @key can be a single string or a tuple (key, suffix)
        """
        if isinstance(key, tuple):
            key, sfx = key
            kidx = self._key_index(key)
            if kidx < 0:
                return False
            return sfx in self._get_suffixes(kidx)

        # assume key is just a string
        return self._has_key(key)

    def __getitem__(self, index):
        """Retrieve an item from the configuration

        If @index is a string, assumes its a key and retrieves the suffixes
        for that key
        if @index is an integer value, return the key and the suffixes for
        that index
        An IndexError is raised if the item is not found
        """
        if isinstance(index, str):
            index = self._key_index(index)
            if index < 0:
                raise IndexError(f'Key "{index}" not found')
            return self._get_suffixes(index)

        # assume index is an integer
        if index < 0:
            return index + len(self)
        if index >= len(self):
            raise IndexError(
                f"Invalid key index {index}. Valid range is [0, {len(self)})"
            )
        key = self._iget_key(index)
        suffixes = self._get_suffixes(index)
        return key, suffixes

    def _get_suffixes(self, kidx):
        suffix_count = self._ikey_get_suffix_count(kidx)
        return [self._ikey_iget_suffix(kidx, s) for s in range(suffix_count)]

    def items(self):
        index = 0
        while index < len(self):
            yield self[index]
            index += 1

    def keys(self):
        for k, _ in self.items():
            yield k

    def free(self):
        self._free()
