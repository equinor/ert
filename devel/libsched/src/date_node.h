#ifndef __DATE_NODE_H__
#define __DATE_NODE_H__
#include <stdbool.h>
#include <hash.h>
#include <time.h>

typedef struct date_node_struct      date_node_type;
date_node_type      * date_node_alloc_from_DATES_line(const time_t * , int , const char * , const hash_type * ); 
date_node_type      * date_node_alloc_from_TSTEP_line(const time_t * , int , const char * , const hash_type * ); 
date_node_type      * date_node_alloc_ext(bool , time_t , int , const time_t * );
int                   date_node_get_date_nr(const date_node_type * );
void                  date_node_get_report_step(const date_node_type * , time_t , int * );
void                  date_node_fprintf_rate_date(const date_node_type * , const char * , const char *);
void                  date_node_fwrite(const date_node_type * , FILE *);
date_node_type      * date_node_fread_alloc(const time_t * , int , time_t , FILE *, bool *);
void                  date_node_fprintf(const date_node_type * , FILE *, int , time_t , bool *);
void                  date_node_free(date_node_type *);
void                  date_node_free__(void *);
void                  date_node_fprintf_days_line(const date_node_type * , FILE *);
date_node_type      * date_node_copyc(const date_node_type * );
void                * date_node_copyc__(const void * );
time_t                date_node_get_date(const date_node_type * );
time_t                date_node_parse_DATES_line(const char * , const hash_type * );
void                  date_node_get_time_t(const date_node_type * , int , time_t *);
#endif
