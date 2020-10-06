#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'subst_config.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.

import os.path

from cwrap import BaseCClass
from res import ResPrototype
from ecl import EclPrototype
from res.enkf import ConfigKeys
from res.util import SubstitutionList


class SubstConfig(BaseCClass):
    TYPE_NAME = "subst_config"
    _alloc = ResPrototype("void* subst_config_alloc(config_content)", bind=False)
    _alloc_full = ResPrototype("void* subst_config_alloc_full(subst_list)", bind=False)
    _free = ResPrototype("void  subst_config_free(subst_config)")
    _get_subst_list = ResPrototype(
        "subst_list_ref subst_config_get_subst_list( subst_config )"
    )
    _get_num_cpu = EclPrototype("int ecl_util_get_num_cpu(char*)", bind=False)

    def __init__(self, config_content=None, config_dict=None):
        if not ((config_content is not None) ^ (config_dict is not None)):
            raise ValueError(
                "SubstConfig must be instansiated with exactly one of config_content or config_dict"
            )

        if config_dict is not None:
            subst_list = SubstitutionList()

            # DIRECTORY #
            config_directory = config_dict.get(ConfigKeys.CONFIG_DIRECTORY)
            if isinstance(config_directory, str):
                subst_list.addItem(
                    "<CWD>",
                    config_directory,
                    "The current working directory we are running from - the location of the config file.",
                )
                subst_list.addItem(
                    "<CONFIG_PATH>",
                    config_directory,
                    "The current working directory we are running from - the location of the config file.",
                )
            else:
                raise ValueError(
                    "{} must be configured".format(ConfigKeys.CONFIG_DIRECTORY)
                )

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

            # Read num_cpu from Eclipse DATA_FILE
            if ConfigKeys.DATA_FILE in config_dict:
                file_path = os.path.join(
                    config_directory, config_dict[ConfigKeys.DATA_FILE]
                )

                if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
                    num_cpu = self._get_num_cpu(file_path)
                    subst_list.addItem(
                        "<NUM_CPU>",
                        "{}".format(num_cpu),
                        "The number of CPU used for one forward model.",
                    )
                else:
                    raise IOError(
                        "Could not find ECLIPSE data file: {}".format(file_path)
                    )

            c_ptr = self._alloc_full(subst_list)

        else:
            c_ptr = self._alloc(config_content)

        if c_ptr is None:
            raise ValueError("Failed to construct Substonfig instance")

        super(SubstConfig, self).__init__(c_ptr)

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

    def __str__(self):
        return (
            "["
            + ",\n".join(
                [
                    "({}, {}, {})".format(key, value, doc)
                    for key, value, doc in self.subst_list
                ]
            )
            + "]"
        )
