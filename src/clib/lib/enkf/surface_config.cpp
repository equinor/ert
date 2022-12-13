#include <ert/enkf/surface_config.hpp>
#include <string>

struct surface_config_struct {
    std::string base_surface_path;
};

surface_config_type *surface_config_alloc_empty() {
    auto config = new surface_config_type;

    return config;
}

void surface_config_free(surface_config_type *config) { delete config; }

void surface_config_set_base_surface(surface_config_type *config,
                                     const char *base_surface) {
    config->base_surface_path = std::string(base_surface);
}
const char *surface_config_base_surface_path(surface_config_type *config) {
    return config->base_surface_path.c_str();
}
