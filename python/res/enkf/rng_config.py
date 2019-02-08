#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'rng_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

from ecl.util.enums import RngAlgTypeEnum

from res import ResPrototype

class RNGConfig(BaseCClass):

    TYPE_NAME = "rng_config"
    _alloc        = ResPrototype("void* rng_config_alloc(config_content)", bind=False)
    _free         = ResPrototype("void rng_config_free(rng_config)")
    _rng_alg_type = ResPrototype("rng_alg_type_enum rng_config_get_type(rng_config)")
    _load_file    = ResPrototype("char* rng_config_get_seed_load_file(rng_config)")
    _store_file   = ResPrototype("char* rng_config_get_seed_store_file(rng_config)")
    _random_seed  = ResPrototype("char* rng_config_get_random_seed(rng_config)")

    def __init__(self, config_content):
        c_ptr = self._alloc(config_content)

        if c_ptr is None:
            raise ValueError('Failed to construct RNGConfig instance')

        super(RNGConfig, self).__init__(c_ptr)

    @property
    def alg_type(self):
        return self._rng_alg_type()

    @property
    def load_filename(self):
        return self._load_file()

    @property
    def store_filename(self):
        return self._store_file()

    @property
    def random_seed(self):
        return self._random_seed()

    def free(self):
        self._free()
