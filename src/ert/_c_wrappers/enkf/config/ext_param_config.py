from typing import Dict, List, Tuple, Union

from ert._c_wrappers.enkf.enums.ert_impl_type_enum import ErtImplType


class ExtParamConfig:
    def __init__(
        self,
        key,
        input_keys: Union[List[str], Dict[str, List[Tuple[str, str]]]],
        output_file: str = "",
        forward_init: bool = False,
        init_file: str = "",
    ):
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

        self.name = key
        self._key_list: List[str] = list(keys)
        self._suffix_list: List[List[str]] = []

        self.output_file: str = output_file
        self.forward_init: bool = forward_init
        self.forward_init_file: str = init_file

        for k, suffixes in suffixmap:
            if not isinstance(suffixes, list):
                raise TypeError(f"Invalid type {type(suffixes)} for suffix: {suffixes}")

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

            self._suffix_list.append(suffixes)

    def __len__(self):
        return len(self._key_list)

    def __contains__(self, key):
        """Check if the @key is present in the configuration

        @key can be a single string or a tuple (key, suffix)
        """

        if isinstance(key, tuple):
            key, suffix = key
            if self._key_list.count(key):
                key_index = self._get_key_index(key)
                return suffix in self._suffix_list[key_index]
            else:
                return False

        # assume key is just a string
        return self._key_list.count(key) > 0

    def __repr__(self):
        return (
            f"SummaryConfig(keylist={self._key_list}), "
            f"suffixlist={self._suffix_list})"
        )

    def __getitem__(self, index):
        """Retrieve an item from the configuration

        If @index is a string, assumes its a key and retrieves the suffixes
        for that key
        if @index is an integer value, return the key and the suffixes for
        that index
        An IndexError is raised if the item is not found
        """
        suffixes = []

        if isinstance(index, str):
            suffix_index = self._get_key_index(index)
            if self._suffix_list_contains_index(suffix_index):
                suffixes = self._key_suffix_value(suffix_index)
            return suffixes

        # assume index is an integer
        if self._suffix_list_contains_index(index):
            suffixes = self._key_suffix_value(index)
        key = self._key_list[index]

        return key, suffixes

    def _get_key_index(self, key: str) -> int:
        if self._key_list.count(key) == 0:
            raise IndexError(
                f"Requested index not found: {key}, " f"Keylist: {self._key_list}"
            )
        return self._key_list.index(key)

    def _suffix_list_contains_index(self, index: int) -> bool:
        return 0 <= index < len(self._suffix_list)

    def _key_suffix_value(self, index: int) -> List[str]:
        if not self._suffix_list_contains_index(index):
            raise IndexError(
                f"Requested index is out of bounds: {index}, "
                f"Suffixlist: {self._suffix_list}"
            )
        return self._suffix_list[index]

    def getKey(self) -> str:
        return self.name

    def getImplementationType(self) -> ErtImplType:  # type: ignore
        return ErtImplType.EXT_PARAM

    def items(self):
        index = 0
        while index < len(self._key_list):
            yield self[index]
            index += 1

    def keys(self):
        for k, _ in self.items():
            yield k
