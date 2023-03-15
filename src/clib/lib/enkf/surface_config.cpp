#include <ert/enkf/surface_config.hpp>
#include <string>

struct surface_config_struct {
    std::string name;
    std::string base_surface_path;
};

surface_config_type *surface_config_alloc(std::string name,
                                          std::string base_surface) {
    auto config = new surface_config_type;
    config->name = name;
    config->base_surface_path = base_surface;
    return config;
}

void surface_config_free(surface_config_type *config) { delete config; }

const char *surface_config_base_surface_path(surface_config_type *config) {
    return config->base_surface_path.c_str();
}

const char *surface_config_name(surface_config_type *config) {
    return config->name.c_str();
}
