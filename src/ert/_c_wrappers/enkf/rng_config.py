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
import struct
from typing import Optional

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._clib.rng_config import log_seed


class RNGConfig(BaseCClass):
    TYPE_NAME = "rng_config"
    _alloc = ResPrototype("void* rng_config_alloc(char*)", bind=False)
    _free = ResPrototype("void rng_config_free(rng_config)")
    _random_seed = ResPrototype("char* rng_config_get_random_seed(rng_config)")

    def __init__(self, random_seed: Optional[str] = None):
        super().__init__(self._alloc(random_seed))

    def __repr__(self):
        return f"RNGConfig(random_seed={self.random_seed})"

    @property
    def random_seed(self):
        return self._random_seed()

    def free(self):
        self._free()

    def __eq__(self, other):
        return self.random_seed == other.random_seed


def format_seed(random_seed: str):
    state_size = 4
    state_digits = 10
    fseed = [0] * state_size
    seed_pos = 0
    for i in range(state_size):
        for k in range(state_digits):
            fseed[i] *= 10
            fseed[i] += ord(random_seed[seed_pos]) - ord("0")
            seed_pos = (seed_pos + 1) % len(random_seed)

    # The function this was derived from had integer overflow, so we
    # allow for the same here
    return b"".join(struct.pack("I", x % (2**32)) for x in fseed)
