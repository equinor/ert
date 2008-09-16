#ifndef __RATE_NODE_H__
#define __RATE_NODE_H__
#include <stdbool.h>
#include <ecl_sum.h>
#include <ecl_well_vars.h>

/*
#define __RATE_ORAT   0
#define __RATE_GRAT   1
#define __RATE_WRAT   2
#define __RATE_GOR    3
#define __RATE_WCT    4
#define __RATE_BHP    5
#define __RATE_THP    6
*/

/*****************************************************************/

typedef struct rate_struct rate_type;

void         rate_fprintf(const rate_type * , FILE * );
void         rate_sched_fprintf(const rate_type *  , FILE *);
rate_type *  rate_alloc_from_token_list(int , const char **);
rate_type *  rate_alloc_from_summary(bool , const ecl_sum_type * , int , const char * );
void         rate_sched_fwrite(const rate_type *,  FILE *);
rate_type *  rate_sched_fread_alloc(FILE *);
void         rate_sched_fprintf_rates(const rate_type * , FILE *);
void         rate_free(rate_type *);
void         rate_free__(void *__rate);
const char * rate_get_well_ref(const rate_type * );
const void * rate_copyc__(const void *);
const char * rate_node_get_well_ref(const rate_type * );

double 	   rate_get_ORAT(const rate_type * , bool *);
double 	   rate_get_GRAT(const rate_type * , bool *);
double 	   rate_get_WRAT(const rate_type * , bool *);
double 	   rate_get_GOR(const rate_type * , bool *, bool * );
double 	   rate_get_WCT(const rate_type * , bool *, bool * );
double     rate_get_THP(const rate_type * , bool *);
double     rate_get_BHP(const rate_type * , bool *);
double     rate_iget(const rate_type * , well_var_type , bool *, bool *);

#endif

