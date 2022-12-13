#ifndef ERT_SURFACE_CONFIG_H
#define ERT_SURFACE_CONFIG_H
#include <ert/enkf/enkf_macros.hpp>
typedef struct surface_config_struct surface_config_type;

surface_config_type *surface_config_alloc_empty();
void surface_config_free(surface_config_type *config);

extern "C" const char *
surface_config_base_surface_path(surface_config_type *config);

void surface_config_set_base_surface(surface_config_type *config,
                                     const char *base_surface);

#endif
