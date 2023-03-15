#ifndef ERT_SURFACE_CONFIG_H
#define ERT_SURFACE_CONFIG_H
#include <ert/enkf/enkf_macros.hpp>
typedef struct surface_config_struct surface_config_type;

surface_config_type *surface_config_alloc(std::string name,
                                          std::string base_surface);
void surface_config_free(surface_config_type *config);

extern "C" const char *
surface_config_base_surface_path(surface_config_type *config);

extern "C" const char *surface_config_name(surface_config_type *config);

#endif
