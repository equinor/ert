#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'gen_kw_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
from cwrap import BaseCClass
import os
from res import ResPrototype
from ecl.util.util import StringList


class GenKwConfig(BaseCClass):
    TYPE_NAME = "gen_kw_config"

    _free = ResPrototype("void  gen_kw_config_free( gen_kw_config )")
    _alloc_empty = ResPrototype(
        "void* gen_kw_config_alloc_empty( char*, char* )", bind=False
    )
    _get_template_file = ResPrototype(
        "char* gen_kw_config_get_template_file(gen_kw_config)"
    )
    _set_template_file = ResPrototype(
        "void  gen_kw_config_set_template_file(gen_kw_config , char*)"
    )
    _get_parameter_file = ResPrototype(
        "char* gen_kw_config_get_parameter_file(gen_kw_config)"
    )
    _set_parameter_file = ResPrototype(
        "void  gen_kw_config_set_parameter_file(gen_kw_config, char*)"
    )
    _alloc_name_list = ResPrototype(
        "stringlist_obj gen_kw_config_alloc_name_list(gen_kw_config)"
    )
    _should_use_log_scale = ResPrototype(
        "bool  gen_kw_config_should_use_log_scale(gen_kw_config, int)"
    )
    _get_key = ResPrototype("char* gen_kw_config_get_key(gen_kw_config)")
    _get_tag_fmt = ResPrototype("char* gen_kw_config_get_tag_fmt(gen_kw_config)")
    _size = ResPrototype("int   gen_kw_config_get_data_size(gen_kw_config)")
    _iget_name = ResPrototype("char* gen_kw_config_iget_name(gen_kw_config, int)")
    _get_function_type = ResPrototype(
        "char* gen_kw_config_iget_function_type(gen_kw_config, int)"
    )
    _get_function_parameter_names = ResPrototype(
        "stringlist_ref gen_kw_config_iget_function_parameter_names(gen_kw_config, int)"
    )
    _get_function_parameter_values = ResPrototype(
        "double_vector_ref gen_kw_config_iget_function_parameter_values(gen_kw_config, int)"
    )

    def __init__(self, key, template_file, parameter_file, tag_fmt="<%s>"):
        """
        @type key: str
        @type tag_fmt: str
        """
        if not os.path.isfile(template_file):
            raise IOError("No such file:%s" % template_file)

        if not os.path.isfile(parameter_file):
            raise IOError("No such file:%s" % parameter_file)

        c_ptr = self._alloc_empty(key, tag_fmt)
        if c_ptr:
            super(GenKwConfig, self).__init__(c_ptr)
        else:
            raise ValueError(
                'Could not instantiate GenKwConfig with key="%s" and tag_fmt="%s"'
                % (key, tag_fmt)
            )
        self._set_parameter_file(parameter_file)
        self._set_template_file(template_file)
        self.__str__ = self.__repr__

    def getTemplateFile(self):
        return self._get_template_file()

    def getParameterFile(self):
        return self._get_parameter_file()

    def getKeyWords(self):
        """ @rtype: StringList """
        return self._alloc_name_list()

    def shouldUseLogScale(self, index):
        """ @rtype: bool """
        return self._should_use_log_scale(index)

    def free(self):
        self._free()

    def __repr__(self):
        return 'GenKwConfig(key = "%s", tag_fmt = "%s") at 0x%x' % (
            self.getKey(),
            self.tag_fmt,
            self._address(),
        )

    def getKey(self):
        """ @rtype: str """
        return self._get_key()

    @property
    def tag_fmt(self):
        return self._get_tag_fmt()

    def __len__(self):
        return self._size()

    def __getitem__(self, index):
        """ @rtype: str """
        return self._iget_name(index)

    def __iter__(self):
        index = 0
        while index < len(self):
            yield self[index]
            index += 1

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        """ @rtype: bool"""
        if self.getTemplateFile() != other.getTemplateFile():
            return False

        if self.getParameterFile() != other.getParameterFile():
            return False

        if self.getKey() != other.getKey():
            return False

        return True

    def get_priors(self):
        """
        @rtype: list
        [
            {
                "key" : "<key>",
                "function" : "<function_type>"
                "parameters" : {
                    "<name>" : "<value>"
                }
            }
        ]
        """
        priors = []
        keys = self.getKeyWords()
        for i, key in enumerate(keys):
            function_type = self._get_function_type(i)
            parameter_names = self._get_function_parameter_names(i)
            parameter_values = self._get_function_parameter_values(i)
            el = {
                "key": key,
                "function": function_type,
                "parameters": {
                    name: value
                    for (name, value) in zip(parameter_names, parameter_values)
                },
            }
            priors.append(el)
        return priors
