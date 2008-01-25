#include <stdlib.h>
#include <stdio.h>
#include <field_config.h>
#include <enkf_types.h>
#include <field.h>
#include <ecl_kw.h>
#include <fortio.h>
#include <ecl_grid.h>
#include <pgbox_config.h>
#include <pgbox.h>
#include <void_arg.h>


double pgfilter(const double *data, const void_arg_type * arg) {
  double x = data[0];
  double y = data[1];

  return x + y;
}


int main (int argc , char **argv) {
  const char *EGRID_file = "OEAST-0025.EGRID";
  int nx , ny , nz , active_size;

  field_config_type * field_config;
  field_type * field              ;
  pgbox_config_type * pg_config   ;
  pgbox_type        * pg;

  ecl_grid_type * ecl_grid         = ecl_grid_alloc_EGRID(EGRID_file , true);
  const int * index_map            = ecl_grid_alloc_index_map(ecl_grid);
  ecl_grid_get_dims(ecl_grid , &nx , &ny , &nz , &active_size);
  field_config = field_config_alloc("KW" , ecl_double_type , nx , ny , nz , active_size , index_map , 0);
  field        = field_alloc(field_config);
  pg_config    = pgbox_config_alloc(ecl_grid , pgfilter , NULL , 2 , 1 , nx , 1 , ny , 1 , 10 , field_config);
  pg           = pgbox_alloc(pg_config);
  pgbox_set_target_field(pg , field);
  pgbox_apply(pg);
  


  pgbox_free(pg);
  pgbox_config_free(pg_config);
  field_config_free(field_config);
  ecl_grid_free(ecl_grid);
  field_free(field);
  free((int *) index_map);
}
