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


     
typedef enum {
  SCHEDULE          = 0,
  REFCASE_SIMULATED = 1,   /* ecl_sum_get_well_var( "WWCT" );  */
  REFCASE_HISTORY   = 2    /* ecl_sum_get_well_var( "WWCTH" ); */
} history_source_type;



typedef struct history_struct history_type;


// Manipulators.
void           history_free(history_type *);
void           history_fwrite(const history_type *, FILE * stream);
history_type * history_fread_alloc(FILE * stream);
history_type * history_alloc_from_sched_file(const sched_file_type *);
void           history_realloc_from_summary(history_type *, const char * refcase , bool );



// Accessors.
void           history_fprintf(const history_type *  , FILE * );
int    	       history_get_num_restarts(const history_type *);
bool           history_str_is_well_name(const history_type *, int, const char *);
bool           history_str_is_group_name(const history_type *, int, const char *);
double         history_get_var_from_sum_key(const history_type *, int, const char *, bool *);
double 	       history_get_well_var(const history_type * , int, const char *, const char *, bool *);
double 	       history_get_group_var(const history_type *, int, const char *, const char *, bool *);
void           history_alloc_time_series_from_summary_key(const history_type *, const char *, int *, double **, bool **);
time_t         history_iget_node_start_time(const history_type *, int);
time_t         history_iget_node_end_time(const history_type *, int);
int            history_get_restart_nr_from_time_t(const history_type *, time_t);
int            history_get_restart_nr_from_days(const history_type *, double days);
char **        history_alloc_well_list(const history_type * , int * );
#ifdef __cplusplus
}
#endif
#endif
