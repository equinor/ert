#  Copyright (C) 2013  Equinor ASA, Norway.
#
#  The file 'config_parser.py' is part of ERT - Ensemble based Reservoir Tool.
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

import sys
import os.path

from cwrap import BaseCClass
from res import ResPrototype
from res.config import ConfigContent, UnrecognizedEnum


class ConfigParser(BaseCClass):
    TYPE_NAME = "config_parser"

    _alloc = ResPrototype("void* config_alloc()", bind=False)
    _add = ResPrototype(
        "schema_item_ref config_add_schema_item(config_parser, char*, bool)"
    )
    _free = ResPrototype("void config_free(config_parser)")
    _parse = ResPrototype(
        "config_content_obj config_parse(config_parser, char*, char*, char*, char*, hash, config_unrecognized_enum, bool)"
    )
    _size = ResPrototype("int config_get_schema_size(config_parser)")
    _get_schema_item = ResPrototype(
        "schema_item_ref config_get_schema_item(config_parser, char*)"
    )
    _has_schema_item = ResPrototype("bool config_has_schema_item(config_parser, char*)")
    _add_key_value = ResPrototype(
        "bool config_parser_add_key_values(config_parser, config_content, char*, stringlist, config_path_elm, char*, config_unrecognized_enum)"
    )
    _validate = ResPrototype("void config_validate(config_parser, config_content)")

    def __init__(self):
        c_ptr = self._alloc()
        super(ConfigParser, self).__init__(c_ptr)

    def __contains__(self, keyword):
        return self._has_schema_item(keyword)

    def __len__(self):
        return self._size()

    def __repr__(self):
        return self._create_repr("size=%d" % len(self))

    def add(self, keyword, required=False, value_type=None):
        item = self._add(keyword, required).setParent(self)
        if value_type:
            item.iset_type(0, value_type)
        return item

    def __getitem__(self, keyword):
        if keyword in self:
            item = self._get_schema_item(keyword)
            item.setParent(self)
            return item
        else:
            raise KeyError("Config parser does not have item:%s" % keyword)

    def parse(
        self,
        config_file,
        comment_string="--",
        include_kw="INCLUDE",
        define_kw="DEFINE",
        pre_defined_kw_map=None,
        unrecognized=UnrecognizedEnum.CONFIG_UNRECOGNIZED_WARN,
        validate=True,
    ):
        """@rtype: ConfigContent"""

        assert isinstance(unrecognized, UnrecognizedEnum)

        if not os.path.exists(config_file):
            raise IOError("File: %s does not exists" % config_file)
        config_content = self._parse(
            config_file,
            comment_string,
            include_kw,
            define_kw,
            pre_defined_kw_map,
            unrecognized,
            validate,
        )
        config_content.setParser(self)

        if validate and not config_content.isValid():
            sys.stderr.write("Errors parsing:%s \n" % config_file)
            for count, error in enumerate(config_content.getErrors()):
                sys.stderr.write("  %02d:%s\n" % (count, error))
            raise ValueError("Parsing:%s failed" % config_file)

        return config_content

    def free(self):
        self._free()

    def validate(self, config_content):
        self._validate(config_content)

    def add_key_value(
        self,
        config_content,
        key,
        value,
        path_elm=None,
        config_filename=None,
        unrecognized_action=UnrecognizedEnum.CONFIG_UNRECOGNIZED_WARN,
    ):

        return self._add_key_value(
            config_content, key, value, path_elm, config_filename, unrecognized_action
        )
