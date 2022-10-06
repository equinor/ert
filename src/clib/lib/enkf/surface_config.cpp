#include <ert/util/util.h>

#include <ert/geometry/geo_surface.h>

#include <ert/enkf/surface_config.hpp>

#define SURFACE_CONFIG_TYPE_ID 853317

struct surface_config_struct {
    UTIL_TYPE_ID_DECLARATION;
    geo_surface_type *base_surface;
};

surface_config_type *surface_config_alloc_empty() {
    surface_config_type *config =
        (surface_config_type *)util_malloc(sizeof *config);
    UTIL_TYPE_ID_INIT(config, SURFACE_CONFIG_TYPE_ID);
    config->base_surface = NULL;
    return config;
}

void surface_config_free(surface_config_type *config) {
    if (config->base_surface != NULL)
        geo_surface_free(config->base_surface);

    free(config);
}

void surface_config_set_base_surface(surface_config_type *config,
                                     const char *base_surface) {
    if (config->base_surface != NULL)
        geo_surface_free(config->base_surface);
    config->base_surface = geo_surface_fload_alloc_irap(base_surface, false);
}

const geo_surface_type *
surface_config_get_base_surface(const surface_config_type *config) {
    return config->base_surface;
}

int surface_config_get_data_size(const surface_config_type *config) {
    return geo_surface_get_size(config->base_surface);
}

void surface_config_ecl_write(const surface_config_type *config,
                              const char *filename, const double *zcoord) {
    geo_surface_fprintf_irap_external_zcoord(config->base_surface, filename,
                                             zcoord);
}

UTIL_SAFE_CAST_FUNCTION(surface_config, SURFACE_CONFIG_TYPE_ID)
UTIL_SAFE_CAST_FUNCTION_CONST(surface_config, SURFACE_CONFIG_TYPE_ID)
VOID_GET_DATA_SIZE(surface)
VOID_CONFIG_FREE(surface)
