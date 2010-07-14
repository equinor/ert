#ifndef __GROUP_RATE_H__
#define __GROUP_RATE_H__

#include <time_t_vector.h>
#include <well_rate.h>
#include <sched_types.h>
#include <sched_history.h>

typedef struct group_rate_struct group_rate_type;
group_rate_type     * group_rate_alloc(const sched_history_type * sched_history , const time_t_vector_type * time_vector , const char * name , const char * phase, const char * type_string , const char * filename);
void                  group_rate_free__( void * arg );
void                  group_rate_add_well_rate( group_rate_type * group_rate , well_rate_type * well_rate);
sched_phase_enum      group_rate_get_phase( const group_rate_type * group_rate );
void                  group_rate_sample( group_rate_type * group_rate );
void                  group_rate_update_wconhist( group_rate_type * group_rate , sched_kw_wconhist_type * kw, int restart_nr );
void                  group_rate_update_wconinje( group_rate_type * group_rate , sched_kw_wconinje_type * kw, int restart_nr );
bool                  group_rate_is_producer( const group_rate_type * group_rate );


#endif
