#ifndef __ECL_CONFIG_H__
#define __ECL_CONFIG_H__
#include <config.h>
#include <ecl_grid.h>
#include <time.h>
#include <ecl_io_config.h>
#include <sched_file.h>
#include <path_fmt.h>


typedef struct ecl_config_struct ecl_config_type;

void                  ecl_config_set_data_file( ecl_config_type * ecl_config , const char * data_file);
ecl_config_type     * ecl_config_alloc( const config_type * );
void                  ecl_config_free( ecl_config_type *);
bool                  ecl_config_include_static_kw(const ecl_config_type * , const char * );
void                  ecl_config_add_static_kw(ecl_config_type *, const char *); 
ecl_io_config_type  * ecl_config_get_io_config(const ecl_config_type * );
sched_file_type     * ecl_config_get_sched_file(const ecl_config_type * );
char                * ecl_config_alloc_schedule_prediction_file(const ecl_config_type * , int );
bool 		      ecl_config_get_formatted(const ecl_config_type * );
bool 		      ecl_config_get_unified_restart(const ecl_config_type * );
bool 		      ecl_config_get_unified_summary(const ecl_config_type * );
const char          * ecl_config_get_data_file(const ecl_config_type * );
const char          * ecl_config_get_schedule_target(const ecl_config_type * );
const char          * ecl_config_get_equil_init_file(const ecl_config_type * );
const path_fmt_type * ecl_config_get_eclbase_fmt(const ecl_config_type * );
int                   ecl_config_get_num_restart_files(const ecl_config_type * );
int                   ecl_config_get_prediction_length(const ecl_config_type * ecl_config );
const ecl_grid_type * ecl_config_get_grid(const ecl_config_type * );
int                   ecl_config_get_last_history_restart( const ecl_config_type * );


#endif
