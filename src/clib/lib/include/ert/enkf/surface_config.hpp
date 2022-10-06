#ifndef ERT_SURFACE_CONFIG_H
#define ERT_SURFACE_CONFIG_H

#include <ert/geometry/geo_surface.hpp>

#include <ert/enkf/enkf_macros.hpp>

typedef struct surface_config_struct surface_config_type;

void surface_config_ecl_write(const surface_config_type *config,
                              const char *filename, const double *zcoord);
const geo_surface_type *
surface_config_get_base_surface(const surface_config_type *config);
void surface_config_free(surface_config_type *config);
int surface_config_get_data_size(const surface_config_type *config);
surface_config_type *surface_config_alloc_empty();
void surface_config_set_base_surface(surface_config_type *config,
                                     const char *base_surface);

UTIL_SAFE_CAST_HEADER(surface_config);
UTIL_SAFE_CAST_HEADER_CONST(surface_config);
GET_DATA_SIZE_HEADER(surface);
VOID_GET_DATA_SIZE_HEADER(surface);
VOID_CONFIG_FREE_HEADER(surface);

#endif
