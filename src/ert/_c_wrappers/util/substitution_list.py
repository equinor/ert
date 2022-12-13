import copy
import os

from cwrap import BaseCClass
from ecl.ecl_util import get_num_cpu as get_num_cpu_from_data_file

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
    _filter_file = ResPrototype("bool subst_list_filter_file(subst_list, char*,char*)")
    _add_from_string = ResPrototype(
        "void subst_list_add_from_string(subst_list, char*)"
    )
    _deep_copy = ResPrototype("subst_list_obj subst_list_alloc_deep_copy(subst_list)")

    def __init__(self):
        c_ptr = self._alloc(None)

        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError("Failed to construct subst_list instance.")

    @classmethod
    def from_dict(cls, config_dict):
        subst_list = SubstitutionList()

        for key, val in config_dict.get("DEFINE", []):
            subst_list.addItem(key, val)
        for key, val in config_dict.get("DATA_KW", []):
            subst_list.addItem(key, val)

        if "<CONFIG_PATH>" not in subst_list:
            config_dir = config_dict.get("CONFIG_DIRECTORY", os.getcwd())
            subst_list.addItem("<CONFIG_PATH>", config_dir)
        else:
            config_dir = subst_list["<CONFIG_PATH>"]

        subst_list.addItem("<RUNPATH_FILE>", config_dict.get("RUNPATH_FILE"))

        num_cpus = config_dict.get("NUM_CPU")
        if num_cpus is None and "DATA_FILE" in config_dict:
            num_cpus = get_num_cpu_from_data_file(config_dict.get("DATA_FILE"))
        if num_cpus is None:
            num_cpus = 1
        subst_list.addItem("<NUM_CPU>", str(num_cpus))

        return subst_list

    def __len__(self):
        return self._size()

    def addItem(self, key, value):
        self._append_copy(key, value)

    def keys(self):
        key_list = []
        for i in range(len(self)):
            key_list.append(self._iget_key(i))
        return key_list

    def __deepcopy__(self, memo):
        return self._deep_copy()

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

    def add_from_string(self, string):
        self._add_from_string(string)

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

    def substitute_file(self, to_substitute: str, tmp_file: str):
        self._filter_file(to_substitute, tmp_file)

    def substitute_real_iter(
        self, to_substitute: str, realization: int, iteration: int
    ) -> str:
        copy_substituter = copy.deepcopy(self)
        geo_id_key = f"<GEO_ID_{realization}_{iteration}>"
        if geo_id_key in self:
            copy_substituter.addItem("<GEO_ID>", self[geo_id_key])
        copy_substituter.addItem("<IENS>", str(realization))
        copy_substituter.addItem("<ITER>", str(iteration))
        return copy_substituter.substitute(to_substitute)

    def __eq__(self, other):
        if self.keys() != other.keys():
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
