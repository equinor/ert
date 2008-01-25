#ifndef __HISTORY_H__
#define __HISTORY_H__
#include <time.h>
#include <stdbool.h>
#include <sched_file.h>
#include <ecl_sum.h>
#include <ecl_well_vars.h>
#include "date_node.h"
#include "rate_node.h"


typedef struct history_struct history_type;



void             history_free(history_type *);
history_type * 	 history_alloc(time_t);
history_type *   history_fread_alloc(FILE *);
history_type * 	 history_alloc_from_schedule(const sched_file_type * );
history_type *   history_alloc_from_summary(const ecl_sum_type *  , bool );
void   history_add_date(history_type *  , const date_node_type * );
void   history_add_rate(history_type *  , int , const rate_type * );
void   history_fwrite(const history_type * , FILE *);

double history_get_ORAT(const history_type * , int , const char * ,  bool *);
double history_get_WRAT(const history_type * , int , const char * ,  bool *);
double history_get_GRAT(const history_type * , int , const char * ,  bool *);
double history_get_THP(const history_type * , int , const char * ,  bool *);
double history_get_BHP(const history_type * , int , const char * ,  bool *);
double history_get_GOR(const history_type * , int , const char * , bool *, bool *);
double history_get_WCT(const history_type * , int , const char * , bool *, bool *);

time_t 		history_get_report_date(history_type * , int );
double 		history_get2(const history_type * , int , const char * , const char * , bool *);
double 	        history_get(const history_type * , int , const char * , const char * );
bool   	        history_has_well(const history_type * , const char * );
/*well_var_type   history_get_var_type(const history_type *  , const char * );   */
history_type     * history_alloc_from_schedule(const sched_file_type *);
int                history_get_num_reports(const history_type * );

#endif
