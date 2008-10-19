#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <gen_param.h>
#include <gen_param_config.h>
#include <util.h>
#include <ecl_grid.h>
#include <pilot_point_config.h>
#include <pilot_point.h>


int main(void) {
  ecl_grid_type           * ecl_grid = ecl_grid_alloc("Gurbat/EXAMPLE_01_BASE.EGRID" , true);
  pilot_point_config_type * config   = pilot_point_config_fscanf_alloc("Gurbat/pilot.conf" , ecl_grid);


  pilot_point_config_free(config);
  ecl_grid_free(ecl_grid);
}



