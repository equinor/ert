import os.path
from typing import Optional

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.util import SubstitutionList


class SubstConfig(BaseCClass):
    TYPE_NAME = "subst_config"
    _alloc = ResPrototype("void* subst_config_alloc(config_content,int)", bind=False)
    _alloc_full = ResPrototype("void* subst_config_alloc_full(subst_list)", bind=False)
    _free = ResPrototype("void  subst_config_free(subst_config)")
    _get_subst_list = ResPrototype(
        "subst_list_ref subst_config_get_subst_list( subst_config )"
    )

    def __init__(
        self, config_content=None, config_dict=None, num_cpu: Optional[int] = None
    ):
        if not (config_content is not None) ^ (config_dict is not None):
            raise ValueError(
                "SubstConfig must be instansiated with exactly one"
                " of config_content or config_dict"
            )

        if config_dict is not None:
            subst_list = SubstitutionList()

            # DIRECTORY #
            config_directory = config_dict.get(ConfigKeys.CONFIG_DIRECTORY)
            if isinstance(config_directory, str):
                subst_list.addItem(
                    "<CWD>",
                    config_directory,
                    "The current working directory we are running from"
                    " - the location of the config file.",
                )
                subst_list.addItem(
                    "<CONFIG_PATH>",
                    config_directory,
                    "The current working directory we are running from"
                    " - the location of the config file.",
                )
            else:
                raise ValueError(f"{ConfigKeys.CONFIG_DIRECTORY} must be configured")

            # FILE #
            filename = config_dict.get(ConfigKeys.CONFIG_FILE_KEY)
            if isinstance(filename, str):
                subst_list.addItem("<CONFIG_FILE>", filename)
                subst_list.addItem("<CONFIG_FILE_BASE>", os.path.splitext(filename)[0])

            # CONSTANTS #
            constants = config_dict.get(ConfigKeys.DEFINE_KEY)
            if isinstance(constants, dict):
                for key in constants:
                    subst_list.addItem(key, constants[key])

            # DATA_KW
            data_kw = config_dict.get(ConfigKeys.DATA_KW_KEY)
            if isinstance(data_kw, dict):
                for key, value in data_kw.items():
                    subst_list.addItem(key, value)

            # RUNPATH_FILE #
            runpath_file_name = config_dict.get(
                ConfigKeys.RUNPATH_FILE, ConfigKeys.RUNPATH_LIST_FILE
            )
            runpath_file_path = os.path.normpath(
                os.path.join(config_directory, runpath_file_name)
            )
            subst_list.addItem(
                "<RUNPATH_FILE>",
                runpath_file_path,
                "The name of a file with a list of run directories.",
            )
            if num_cpu is not None:
                subst_list.addItem(
                    "<NUM_CPU>",
                    str(num_cpu),
                    "The number of CPU used for one forward model.",
                )

            c_ptr = self._alloc_full(subst_list)

        else:
            num_cpu_as_int = num_cpu if num_cpu is not None else 0
            c_ptr = self._alloc(config_content, num_cpu_as_int)

        if c_ptr is None:
            raise ValueError("Failed to construct Substonfig instance")

        super().__init__(c_ptr)

    def __getitem__(self, key):
        subst_list = self._get_subst_list()
        return subst_list[key]

    def __iter__(self):
        subst_list = self._get_subst_list()
        return iter(subst_list)

    @property
    def subst_list(self):
        return self._get_subst_list().setParent(self)

    def free(self):
        self._free()

    def __eq__(self, other):
        list1 = self.subst_list
        list2 = other.subst_list
        if len(list1.keys()) != len(list2.keys()):
            return False
        for key in list1.keys():
            val1 = list1.get(key)
            val2 = list2.get(key)
            if val1 != val2:
                return False

        return True

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return f"SubstConfig({str(self)})"

    def __str__(self):
        if not self._address():
            return ""
        return (
            "["
            + ",\n".join(
                [f"({key}, {value}, {doc})" for key, value, doc in self.subst_list]
            )
            + "]"
        )
