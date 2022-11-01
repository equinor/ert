from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype


class SubstitutionList(BaseCClass):
    TYPE_NAME = "subst_list"

    _alloc = ResPrototype("void* subst_list_alloc()", bind=False)
    _free = ResPrototype("void subst_list_free(subst_list)")
    _size = ResPrototype("int subst_list_get_size(subst_list)")
    _iget_key = ResPrototype("char* subst_list_iget_key(subst_list, int)")
    _get_value = ResPrototype("char* subst_list_get_value(subst_list, char*)")
    _has_key = ResPrototype("bool subst_list_has_key(subst_list, char*)")
    _append_copy = ResPrototype("void subst_list_append_copy(subst_list, char*, char*)")
    _alloc_filtered_string = ResPrototype(
        "char* subst_list_alloc_filtered_string(subst_list, char*)"
    )

    def __init__(self):
        c_ptr = self._alloc(None)

        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError("Failed to construct subst_list instance.")

    def __len__(self):
        return self._size()

    def addItem(self, key, value):
        self._append_copy(key, value)

    def keys(self):
        key_list = []
        for i in range(len(self)):
            key_list.append(self._iget_key(i))
        return key_list

    def __iter__(self):
        index = 0
        keys = self.keys()
        for index in range(len(self)):
            key = keys[index]
            yield (key, self[key])

    def __contains__(self, key):
        if not isinstance(key, str):
            return False
        return self._has_key(key)

    def __getitem__(self, key):
        if key in self:
            return self._get_value(key)
        else:
            raise KeyError(f"No such key:{key}")

    def get(self, key, default=None):
        return self[key] if key in self else default

    def indexForKey(self, key):
        if key not in self:
            raise KeyError(f"Key '{key}' not in substitution list!")

        for index, key_value in enumerate(self):
            if key == key_value[0]:
                return index

        return None  # Should never happen!

    def substitute(self, to_substitute: str) -> str:
        return self._alloc_filtered_string(to_substitute)

    def __eq__(self, other):
        if len(self.keys()) != len(other.keys()):
            return False
        for key in self.keys():
            oneValue = self.get(key)
            otherValue = other.get(key)
            if oneValue != otherValue:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def free(self):
        self._free()

    def _concise_representation(self):
        return (
            ("[" + ",\n".join([f"({key}, {value})" for key, value in self]) + "]")
            if self._address()
            else ""
        )

    def __repr__(self):
        return f"<SubstitutionList({self._concise_representation()})>"

    def __str__(self):
        return f"SubstitutionList({self._concise_representation()})"
