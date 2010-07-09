#ifndef __SCHED_HISTORY_H__
#define __SCHED_HISTORY_H__

#ifdef __cplusplus 
extern "C" {
#endif

#include <sched_file.h>

typedef struct sched_history_struct sched_history_type;


void                  sched_history_update( sched_history_type * sched_history, const sched_file_type * sched_file );
sched_history_type *  sched_history_alloc( const char * sep_string );
void                  sched_history_free( sched_history_type * sched_history );
double                sched_history_iget( const sched_history_type * sched_history , const char * key , int report_step);



#ifdef __cplusplus 
}
#endif

#endif
