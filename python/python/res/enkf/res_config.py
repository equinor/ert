#  Copyright (C) 2017  Statoil ASA, Norway. 
#   
#  The file 'res_config.py' is part of ERT - Ensemble based Reservoir Tool. 
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

from os.path import isfile

from cwrap import BaseCClass
from res.enkf import EnkfPrototype
from res.enkf import SiteConfig
from res.enkf import SubstConfig

class ResConfig(BaseCClass):

    TYPE_NAME = "res_config"

    _alloc = EnkfPrototype("void* res_config_alloc_load(char*)", bind=False)
    _free  = EnkfPrototype("void res_config_free(res_config)")

    _user_config_file = EnkfPrototype("char* res_config_get_user_config_file(res_config)")

    _config_path      = EnkfPrototype("char* res_config_get_working_directory(res_config)")
    _site_config      = EnkfPrototype("site_config_ref res_config_get_site_config(res_config)")
    _analysis_config  = EnkfPrototype("analysis_config_ref res_config_get_analysis_config(res_config)")
    _subst_config     = EnkfPrototype("subst_config_ref res_config_get_subst_config(res_config)")

    def __init__(self, user_config_file):
        if user_config_file is not None and not isfile(user_config_file):
            raise IOError('No such configuration file "%s".' % user_config_file)

        c_ptr = self._alloc(user_config_file)
        if c_ptr:
            super(ResConfig, self).__init__(c_ptr)
        else:
            raise ValueError(
                    'Failed to construct EnkfConfig instance from config file %s.'
                    % user_config_file
                    )

    def free(self):
        self._free()

    @property
    def user_config_file(self):
        return self._user_config_file()

    @property
    def site_config_file(self):
        return self.site_config.config_file

    @property
    def site_config(self):
        return self._site_config()

    @property
    def analysis_config(self):
        return self._analysis_config()

    @property
    def config_path(self):
        return self._config_path( )

    @property
    def subst_config(self):
        return self._subst_config( )
