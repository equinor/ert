#ifndef __SCHED_TIME_H__
#define __SCHED_TIME_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <sched_types.h>

typedef struct sched_time_struct sched_time_type;

sched_time_type * sched_time_alloc( time_t date , double tstep_length , sched_time_enum  time_type );
void              sched_time_free( sched_time_type * time_node );
void              sched_time_free__( void * arg );
time_t            sched_time_get_date( const sched_time_type * time_node );
time_t            sched_time_get_type( const sched_time_type * time_node );
time_t            sched_time_get_target( const sched_time_type * time_node , time_t current_time);

#ifdef __cplusplus
}
#endif
#endif
