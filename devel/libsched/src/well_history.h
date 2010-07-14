#ifndef __WELL_HISTORY__
#define __WELL_HISTORY__

#ifdef __cplusplus
extern "C" {
#endif

#include <sched_kw.h>
#include <sched_kw_wconhist.h>
#include <size_t_vector.h>
#include <well_index.h>
#include <group_history.h>

typedef struct well_history_struct  well_history_type;




bool                  well_history_is_producer( const well_history_type * well_history , int report_step );
wconhist_state_type * well_history_get_wconhist( well_history_type * well_history );
well_history_type   * well_history_alloc( const char * well_name , const time_t_vector_type * time);
void                  well_history_free__(void * arg);
void                  well_history_add_keyword( well_history_type * well_history, const sched_kw_type * sched_kw , int  report_step );
const void          * well_history_get_state_ptr( const well_history_type * well_history , sched_kw_type_enum kw_type );
const char          * well_history_get_name( const well_history_type * well_history );

sched_kw_type_enum    well_history_iget_active_kw( const well_history_type * history , int report_step );
double                well_history_iget( well_index_type * index , int report_step );
void                  well_history_set_parent( well_history_type * child_well , int report_step , const group_history_type * parent_group);
group_history_type  * well_history_get_parent( well_history_type * child_well , int report_step );


double                well_history_iget_WGPRH( const well_history_type * well_history , int report_step );
double                well_history_iget_WOPRH( const well_history_type * well_history , int report_step );
double                well_history_iget_WWPRH( const well_history_type * well_history , int report_step );

UTIL_IS_INSTANCE_HEADER( well_history )


#ifdef __cplusplus
}
#endif

#endif
