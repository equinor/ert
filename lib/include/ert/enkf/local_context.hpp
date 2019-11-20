/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'local_context.h' is part of ERT - Ensemble based Reservoir Tool.

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


#ifndef ERT_LOCAL_CONTEXT_H
#define ERT_LOCAL_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include <ert/geometry/geo_polygon.h>
#include <ert/geometry/geo_surface.h>
#include <ert/geometry/geo_region.h>

#include <ert/ecl/ecl_region.h>
#include <ert/ecl/ecl_file.h>
#include <ert/ecl/ecl_grid.h>

#define  GLOBAL_GRID  "GLOBAL_GRID"

  typedef struct local_context_struct local_context_type;

#ifdef __cplusplus
}
#endif
#endif
