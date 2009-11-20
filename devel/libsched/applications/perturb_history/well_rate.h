#ifndef __WELL_RATE_H__
#define __WELL_RATE_H__

#include <time_t_vector.h>
#include <sched_file.h>
#include <sched_kw_wconhist.h>
#include <sched_kw_wconinje.h>
#include <sched_types.h>


typedef struct well_rate_struct well_rate_type;
well_rate_type     * well_rate_alloc(const time_t_vector_type * time_vector , 
                                     const sched_file_type    * sched_file  , 
                                     const char * name , double corr_length , const char * phase, const char * filename);
void                 well_rate_free__( void * arg );
double_vector_type * well_rate_get_shift( well_rate_type * well_rate );
sched_phase_type     well_rate_get_phase( const well_rate_type * well_rate );
const char         * well_rate_get_name( const well_rate_type * well_rate );
void                 well_rate_sample_shift( well_rate_type * well_rate );
bool                 well_rate_well_open( const well_rate_type * well_rate , int index );
void                 well_rate_ishift( well_rate_type * well_rate  ,int index, double shift);
void                 well_rate_update_wconhist( well_rate_type * well_rate , sched_kw_wconhist_type * kw, int restart_nr );
void                 well_rate_update_wconinje( well_rate_type * well_rate , sched_kw_wconinje_type * kw, int restart_nr );

#endif
