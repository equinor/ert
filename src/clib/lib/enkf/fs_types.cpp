#include <ert/enkf/fs_types.hpp>

/**
  @brief returns whether fs type is valid.

*/
bool fs_types_valid(fs_driver_enum driver_type) {
    return ((driver_type == DRIVER_PARAMETER) ||
            (driver_type == DRIVER_INDEX) ||
            (driver_type == DRIVER_DYNAMIC_FORECAST));
}
