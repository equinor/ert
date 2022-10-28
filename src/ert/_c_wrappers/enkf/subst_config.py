import os.path
from datetime import date
from typing import Dict

from ert._c_wrappers.config import ConfigContent
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.util import SubstitutionList


class SubstConfig:
    def __init__(
        self,
        defines: Dict[str, str],
        data_kw: Dict[str, str],
        runpath_file_name: str,
        num_cpu: int,
    ):
        subst_list = SubstitutionList()

        today_date_string = date.today().isoformat()
        subst_list.addItem("<DATE>", today_date_string, "The current date.")

        for key, value in defines.items():
            subst_list.addItem(key, value)
        for key, value in data_kw.items():
            subst_list.addItem(key, value)

        if "<CONFIG_PATH>" in defines:
            runpath_file_path = os.path.normpath(
                os.path.join(defines["<CONFIG_PATH>"], runpath_file_name)
            )
        else:
            runpath_file_path = os.path.abspath(runpath_file_name)
        subst_list.addItem(
            "<RUNPATH_FILE>",
            runpath_file_path,
            "The name of a file with a list of run directories.",
        )
        subst_list.addItem(
            "<NUM_CPU>",
            str(num_cpu),
            "The number of CPU used for one forward model.",
        )
        self.subst_list = subst_list

    @classmethod
    def from_dict(cls, config_dict: dict, num_cpu: int):
        init_args = {}
        init_args["defines"] = config_dict.get(ConfigKeys.DEFINE_KEY, {})
        init_args["data_kw"] = config_dict.get(ConfigKeys.DATA_KW_KEY, {})
        init_args["runpath_file_name"] = config_dict.get(
            ConfigKeys.RUNPATH_FILE, ConfigKeys.RUNPATH_LIST_FILE
        )
        init_args["num_cpu"] = num_cpu
        return cls(**init_args)

    @classmethod
    def from_config_content(cls, config_content: ConfigContent, num_cpu: int):
        init_args = {}
        init_args["runpath_file_name"] = (
            config_content.getValue(ConfigKeys.RUNPATH_FILE)
            if ConfigKeys.RUNPATH_FILE in config_content
            else ConfigKeys.RUNPATH_LIST_FILE
        )

        init_args["data_kw"] = {}
        if ConfigKeys.DATA_KW_KEY in config_content:
            for data_kw_definition in config_content[ConfigKeys.DATA_KW_KEY]:
                init_args["data_kw"][data_kw_definition[0]] = data_kw_definition[1]

        init_args["defines"] = dict(
            (key, value) for key, value, _ in config_content.get_const_define_list()
        )
        init_args["num_cpu"] = num_cpu
        return cls(**init_args)

    def __getitem__(self, key):
        return self.subst_list[key]

    def __iter__(self):
        return iter(self.subst_list)

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
        concise_substitution_list = (
            "["
            + ",\n".join([f"({key}, {value})" for key, value, _ in self.subst_list])
            + "]"
        )
        return f"<SubstConfig({concise_substitution_list})>"

    def __str__(self):
        return (
            "["
            + ",\n".join(
                [f"({key}, {value}, {doc})" for key, value, doc in self.subst_list]
            )
            + "]"
        )
