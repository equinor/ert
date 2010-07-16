#ifndef __HISTORY_H__
#define __HISTORY_H__
#ifdef __cplusplus
extern "C" {
#endif


#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <ecl_sum.h>
#include <sched_file.h>
#include <bool_vector.h>
#include <double_vector.h>

     
typedef enum {
  SCHEDULE          = 0,
  REFCASE_SIMULATED = 1,   /* ecl_sum_get_well_var( "WWCT" );  */
  REFCASE_HISTORY   = 2    /* ecl_sum_get_well_var( "WWCTH" ); */
} history_source_type;



typedef struct history_struct history_type;

history_source_type history_get_source_type( const char * string_source );

// Manipulators.
void           history_free(history_type *);
history_type * history_alloc_from_sched_file(const char * sep_string , const sched_file_type *);
void           history_realloc_from_summary(history_type *, const ecl_sum_type * refcase , bool );
const char   * history_get_source_string( history_source_type history_source );
void           history_init_ts( const history_type * history , const char * summary_key , double_vector_type * value, bool_vector_type * valid);

// Accessors.
void           history_fprintf(const history_type *  , FILE * );
int    	       history_get_num_restarts(const history_type *);
double         history_get_var_from_sum_key(const history_type *, int, const char *, bool *);
double 	       history_get_well_var(const history_type * , int, const char *, const char *, bool *);
double 	       history_get_group_var(const history_type *, int, const char *, const char *, bool *);
void           history_alloc_time_series_from_summary_key(const history_type *, const char *, double **, bool **);
time_t         history_iget_node_start_time(const history_type *, int);
time_t         history_iget_node_end_time(const history_type *, int);
int            history_get_restart_nr_from_time_t(const history_type *, time_t);
int            history_get_restart_nr_from_days(const history_type *, double days);
time_t         history_get_time_t_from_restart_nr( const history_type * history , int restart_nr);
char **        history_alloc_well_list(const history_type * , int * );
#ifdef __cplusplus
}
#endif
#endif
