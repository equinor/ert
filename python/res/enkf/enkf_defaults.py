#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'enkf_defaults.py' is part of ERT - Ensemble based Reservoir Tool.
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

from res import ResPrototype


class EnkfDefaults:
    enkf_defaults_get_default_gen_kw_export_name = ResPrototype(
        "char* enkf_defaults_get_default_gen_kw_export_name()", bind=False
    )
    DEFAULT_GEN_KW_EXPORT_NAME = enkf_defaults_get_default_gen_kw_export_name()
