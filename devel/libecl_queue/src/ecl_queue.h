#ifndef __ECL_QUEUE_H__
#define __ECL_QUEUE_H__

typedef struct ecl_queue_struct ecl_queue_type;
ecl_queue_type   * ecl_queue_alloc(int  , int  , const char *  , const char *  , const char *  , const char *  , const char *  , void * );
void               ecl_queue_free(ecl_queue_type *);
void               ecl_queue_add_job(ecl_queue_type * , int );

#endif
