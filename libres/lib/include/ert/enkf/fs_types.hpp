/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'fs_types.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_FS_TYPES_H
#define ERT_FS_TYPES_H

/**
  The various driver implementations - this goes on disk all over the
  place, and the numbers should be considered SET IN STONE. When a new
  driver is added the switch statement in the enkf_fs_mount() function
  must be updated.
*/
typedef enum {
    INVALID_DRIVER_ID = 0,
    BLOCK_FS_DRIVER_ID = 3001
} fs_driver_impl;

/**
  The categories of drivers. To reduce the risk of programming
  error (or at least to detect it ...), there should be no overlap
  between these ID's and the ID's of the actual implementations
  above. The same comment about permanent storage applies to these
  numbers as well.
*/
typedef enum {
    DRIVER_PARAMETER = 1,
    DRIVER_INDEX = 4,
    DRIVER_DYNAMIC_FORECAST = 5,
} fs_driver_enum;

bool fs_types_valid(fs_driver_enum driver_type);

#endif
