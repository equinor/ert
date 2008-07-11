#ifndef __TPGZONE_UTIL_H__
#define __TPGZONE_UTIL_H__

#include <stdbool.h>
#include <ecl_grid.h>
#include <field_config.h>

field_config_type * tpgzone_field_config_alloc__(const char *, const char *, const ecl_grid_type *, bool);

#endif
