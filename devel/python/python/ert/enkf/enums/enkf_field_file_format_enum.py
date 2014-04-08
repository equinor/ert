#  Copyright (C) 2013  Statoil ASA, Norway.
#
#  The file 'content_type_enum.py' is part of ERT - Ensemble based Reservoir Tool.
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
from ert.cwrap import BaseCEnum
from ert.enkf import ENKF_LIB


class EnkfFieldFileFormatEnum(BaseCEnum):
    UNDEFINED_FORMAT         = 0
    RMS_ROFF_FILE            = 1
    ECL_KW_FILE              = 2
    ECL_KW_FILE_ACTIVE_CELLS = 3
    ECL_KW_FILE_ALL_CELLS    = 4
    ECL_GRDECL_FILE          = 5
    ECL_FILE                 = 6
    FILE_FORMAT_NULL         = 7

#EnkfFieldFileFormatEnum.populateEnum(ENKF_LIB, "enkf_field_file_format_iget")
EnkfFieldFileFormatEnum.registerEnum(ENKF_LIB, "enkf_field_file_format_enum")

EnkfFieldFileFormatEnum.INITIALIZATION_TYPES = [EnkfFieldFileFormatEnum.ECL_KW_FILE]



