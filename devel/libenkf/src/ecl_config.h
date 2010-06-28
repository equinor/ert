#ifndef __ECL_CONFIG_H__
#define __ECL_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif


#include <config.h>
#include <ecl_grid.h>
#include <ecl_sum.h>
#include <time.h>
#include <ecl_io_config.h>
#include <sched_file.h>
#include <path_fmt.h>


typedef struct ecl_config_struct ecl_config_type;

const char          * ecl_config_get_schedule_prediction_file( const ecl_config_type * ecl_config );
void                  ecl_config_set_schedule_prediction_file( ecl_config_type * ecl_config , const char * schedule_prediction_file );
const char *          ecl_config_get_schedule_file( const ecl_config_type * ecl_config );
int                   ecl_config_get_num_cpu( const ecl_config_type * ecl_config );
void                  ecl_config_set_data_file( ecl_config_type * ecl_config , const char * data_file);
void                  ecl_config_init( ecl_config_type * ecl_config , const config_type * config);
void                  ecl_config_free( ecl_config_type *);
bool                  ecl_config_include_static_kw(const ecl_config_type * , const char * );
void                  ecl_config_add_static_kw(ecl_config_type *, const char *); 
ecl_io_config_type  * ecl_config_get_io_config(const ecl_config_type * );
sched_file_type     * ecl_config_get_sched_file(const ecl_config_type * );
bool 		      ecl_config_get_formatted(const ecl_config_type * );
bool 		      ecl_config_get_unified_restart(const ecl_config_type * );
bool 		      ecl_config_get_unified_summary(const ecl_config_type * );
const char          * ecl_config_get_data_file(const ecl_config_type * );
const char          * ecl_config_get_schedule_target(const ecl_config_type * );
const char          * ecl_config_get_equil_init_file(const ecl_config_type * );
void                  ecl_config_set_init_section( ecl_config_type * ecl_config , const char * input_init_section );
const char          * ecl_config_get_init_section(const ecl_config_type * ecl_config);
const path_fmt_type * ecl_config_get_eclbase_fmt(const ecl_config_type * );
int                   ecl_config_get_num_restart_files(const ecl_config_type * );
const ecl_sum_type  * ecl_config_get_refcase(const ecl_config_type * ecl_config);
ecl_grid_type       * ecl_config_get_grid(const ecl_config_type * );
void                  ecl_config_set_grid( ecl_config_type * ecl_config , const char * grid_file );
const char          * ecl_config_get_gridfile( const ecl_config_type * ecl_config );
int                   ecl_config_get_last_history_restart( const ecl_config_type * );
bool                  ecl_config_can_restart( const ecl_config_type * ecl_config );
void                  ecl_config_set_eclbase( ecl_config_type * ecl_config , const char * eclbase_fmt );
const char          * ecl_config_get_eclbase( const ecl_config_type * ecl_config );
const char          * ecl_config_get_schedule_file( const ecl_config_type * ecl_config );
void                  ecl_config_set_schedule_file( ecl_config_type * ecl_config , const char * schedule_file );
void                  ecl_config_load_refcase( ecl_config_type * ecl_config , const char * refcase);
const char          * ecl_config_get_refcase_name( const ecl_config_type * ecl_config);
void                  ecl_config_clear_static_kw( ecl_config_type * ecl_config );
stringlist_type     * ecl_config_get_static_kw_list( const ecl_config_type * ecl_config );
void                  ecl_config_fprintf_config( const ecl_config_type * ecl_config , FILE * stream );
ecl_config_type     * ecl_config_alloc_empty( );

#ifdef __cplusplus
}
#endif
#endif
