/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'local_context.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <stdbool.h>

#include <ert/util/util.h>
#include <ert/util/hash.h>

#include <ert/geometry/geo_polygon.h>
#include <ert/geometry/geo_surface.h>
#include <ert/geometry/geo_region.h>

#include <ert/ecl/ecl_grid.h>

#include <ert/enkf/local_context.hpp>

struct local_context_struct {
  hash_type * ecl_regions;
  hash_type * files;
  hash_type * polygons;
  hash_type * grids;
  hash_type * surfaces;
  hash_type * surface_regions;
};


/*************************/

static void local_context_add_polygon__( local_context_type * context , const char * polygon_name , geo_polygon_type * polygon)  {
  hash_insert_hash_owned_ref( context->polygons , polygon_name , polygon , geo_polygon_free__);
}
