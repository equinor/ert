#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'ert_template.py' is part of ERT - Ensemble based Reservoir Tool.
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
from res import ResPrototype


class ErtTemplate(BaseCClass):
    TYPE_NAME = "ert_template"

    _free = ResPrototype("void  ert_template_free( ert_template )")
    _get_template_file = ResPrototype(
        "char* ert_template_get_template_file(ert_template)"
    )
    _get_target_file = ResPrototype("char* ert_template_get_target_file(ert_template)")
    _get_arg_list = ResPrototype(
        "subst_list_ref ert_template_get_arg_list( ert_template )"
    )

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def get_template_file(self):
        """@rtype: str"""
        return self._get_template_file()

    def get_target_file(self):
        """@rtype: str"""
        return self._get_target_file()

    def get_args_as_string(self):
        """@rtype: str"""
        args_list = self._get_arg_list()
        return ", ".join(
            ["{}={}".format(key, args_list.get(key)) for key in args_list.keys()]
        )

    def __eq__(self, other):
        return (
            self.get_template_file() == other.get_template_file()
            and self.get_target_file() == other.get_target_file()
            and self.get_args_as_string() == other.get_args_as_string()
        )

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "ErtTemplate({}, {}, {})".format(
            self.get_template_file(), self.get_target_file(), self.get_args_as_string()
        )

    def free(self):
        self._free()
