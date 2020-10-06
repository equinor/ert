#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'ert_templates.py' is part of ERT - Ensemble based Reservoir Tool.
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
from res.enkf import ErtTemplate, ConfigKeys
from ecl.util.util import StringList
import os


class ErtTemplates(BaseCClass):
    TYPE_NAME = "ert_templates"
    _alloc = ResPrototype(
        "void* ert_templates_alloc( subst_list, config_content )", bind=False
    )
    _alloc_default = ResPrototype(
        "void* ert_templates_alloc_default( subst_list )", bind=False
    )
    _free = ResPrototype("void ert_templates_free( ert_templates )")
    _alloc_list = ResPrototype("stringlist_ref ert_templates_alloc_list(ert_templates)")
    _get_template = ResPrototype(
        "ert_template_ref ert_templates_get_template(ert_templates, char*)"
    )
    _clear = ResPrototype("void ert_templates_clear(ert_templates)")
    _add_template = ResPrototype(
        "ert_template_ref ert_templates_add_template(ert_templates, char*, char*, char*, char*)"
    )
    _add_template_unbound = ResPrototype(
        "ert_template_ref ert_templates_add_template(ert_templates, char*, char*, char*, char*)",
        bind=False,
    )

    def __init__(self, parent_subst, config_content=None, config_dict=None):
        if not ((config_content is not None) ^ (config_dict is not None)):
            raise ValueError(
                "ErtTemplates must be instantiated with exactly one of config_content or config_dict"
            )

        if config_dict is not None:
            c_ptr = self._alloc_default(parent_subst)
            if c_ptr is None:
                raise ValueError("Failed to construct ErtTemplates instance")
            super(ErtTemplates, self).__init__(c_ptr)
            run_template = config_dict.get(ConfigKeys.RUN_TEMPLATE)
            if isinstance(run_template, list):
                for template_file_name, target_file, arguments in run_template:
                    path = config_dict.get(ConfigKeys.CONFIG_DIRECTORY)
                    if not isinstance(path, str):
                        raise ValueError(
                            "ErtTemplates requires {} to be set".format(
                                ConfigKeys.CONFIG_DIRECTORY
                            )
                        )
                    template_path = os.path.normpath(
                        os.path.join(path, template_file_name)
                    )
                    arguments_string = ", ".join(
                        ["{}={}".format(key, val) for key, val in arguments]
                    )
                    self._add_template(
                        None, template_path, target_file, arguments_string
                    )

        else:
            c_ptr = self._alloc(parent_subst, config_content)
            if c_ptr is None:
                raise ValueError("Failed to construct ErtTemplates instance")
            super(ErtTemplates, self).__init__(c_ptr)

    def getTemplateNames(self):
        """ @rtype: StringList """
        return self._alloc_list().setParent(self)

    def clear(self):
        self._clear()

    def get_template(self, key):
        """ @rtype: ErtTemplate """
        return self._get_template(key).setParent(self)

    def add_template(self, key, template_file, target_file, arg_string):
        """ @rtype: ErtTemplate """
        return self._add_template(
            key, template_file, target_file, arg_string
        ).setParent(self)

    def __eq__(self, other):
        if len(self.getTemplateNames()) != len(other.getTemplateNames()):
            return False
        if not all(
            name in self.getTemplateNames() for name in other.getTemplateNames()
        ):
            return False
        for name in self.getTemplateNames():
            if self.get_template(name) != other.get_template(name):
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "ErtTemplates({})".format(
            ", ".join(
                x + "=" + str(self.get_template(x)) for x in self.getTemplateNames()
            )
        )

    def free(self):
        self._free()
