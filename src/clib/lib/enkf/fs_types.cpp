/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'fs_types.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <ert/enkf/fs_types.hpp>

/**
  @brief returns whether fs type is valid.

*/
bool fs_types_valid(fs_driver_enum driver_type) {
    return ((driver_type == DRIVER_PARAMETER) ||
            (driver_type == DRIVER_INDEX) ||
            (driver_type == DRIVER_DYNAMIC_FORECAST));
}
