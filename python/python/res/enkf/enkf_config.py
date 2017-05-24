#  Copyright (C) 2017  Statoil ASA, Norway. 
#   
#  The file 'enkf_config.py' is part of ERT - Ensemble based Reservoir Tool. 
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

class EnkfConfig(BaseCClass):

    TYPE_NAME = "enkf_config"

    _alloc = EnkfPrototype("void* enkf_config_alloc_load(char*)", bind=False)
    _free  = EnkfPrototype("void enkf_config_free(enkf_config)")

    _user_config_file = EnkfPrototype("char* enkf_config_get_user_config_file(enkf_config)")
    _site_config      = EnkfPrototype("site_config_ref enkf_config_get_site_config(enkf_config)")

    def __init__(self, user_config_file):
        if user_config_file is not None and not isfile(user_config_file):
            raise IOError('No such configuration file "%s".' % user_config_file)

        c_ptr = self._alloc(user_config_file)
        if c_ptr:
            super(EnkfConfig, self).__init__(c_ptr)
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
