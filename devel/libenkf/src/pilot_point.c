#include <enkf_types.h>
#include <util.h>
#include <enkf_macros.h>
#include <scalar.h>
#include <hash.h>
#include <pilot_point_config.h>
#include <pilot_point.h>
#include <field.h>


struct pilot_point_struct {
  const pilot_point_config_type * config;
  scalar_type                   * data;
};


/*****************************************************************/


pilot_point_type * pilot_point_alloc( const pilot_point_config_type * config) {
  pilot_point_type * pilot_point = util_malloc( sizeof * pilot_point , __func__);
  pilot_point->config = config;
  
  return pilot_point;
}


void pilot_point_free(pilot_point_type * pp) {
  scalar_free(pp->data);
  free(pp);
}


void pilot_point_apply(const pilot_point_type * pp , field_type * target_field , int target_size, const int * field_index_list , const int * grid_index_list) {
  
}
