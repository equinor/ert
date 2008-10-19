#ifndef __PILOT_POINT_CONFIG__
#define __PILOT_POINT_CONFIG__
#include <scalar_config.h>
#include <ecl_grid.h>

typedef struct pilot_point_config_struct  pilot_point_config_type;


pilot_point_config_type * pilot_point_config_alloc();
void                      pilot_point_config_free(pilot_point_config_type *);
pilot_point_config_type * pilot_point_config_fscanf_alloc(const char * , const ecl_grid_type * );
scalar_config_type      * pilot_point_config_get_scalar_config(const pilot_point_config_type * );

#endif

