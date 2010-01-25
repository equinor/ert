#ifndef __WELL_HISTORY__
#define __WELL_HISTORY__

#ifdef __cplusplus
extern "C" {
#endif

#include <sched_kw_wconhist.h>
#include <sched_kw_wconinjh.h>

typedef struct well_history_struct    well_history_type;



wconhist_state_type * well_history_get_wconhist( well_history_type * well_history );
well_history_type   * well_history_alloc( const char * well_name );
void                  well_history_free__(void * arg);







#ifdef __cplusplus
}
#endif

#endif
