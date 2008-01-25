#ifndef __OBS_NODE__
#define __OBS_NODE__
#include <enkf_macros.h>
#include <stdbool.h>
#include <enkf_types.h>
#include <obs_data.h>
#include <meas_vector.h>
#include <time.h>
#include <sched_file.h>

typedef void (obs_free_ftype)                (void *);
typedef void (obs_get_ftype)                 (const void * , int , obs_data_type *);
typedef void (obs_meas_ftype)                (const void * , const void *, meas_vector_type *);

typedef struct obs_node_struct obs_node_type;


void            obs_node_measure(const obs_node_type *  , int , const void * , meas_vector_type * );
obs_node_type * obs_node_alloc(const void * , const char * , int , bool , obs_get_ftype * , obs_meas_ftype * , obs_free_ftype *);
void            obs_node_free(obs_node_type * );
const void *    obs_node_get_ref(const obs_node_type * );
void            obs_node_get_observations(obs_node_type * , int , obs_data_type * );
void 		obs_node_activate_report_step(obs_node_type * , int , int );
void 		obs_node_deactivate_report_step(obs_node_type * , int , int );
void 		obs_node_activate_time_t(obs_node_type * , const sched_file_type * , time_t , time_t );
void 		obs_node_deactivate_time_t(obs_node_type * , const sched_file_type * , time_t , time_t );



VOID_FREE_HEADER(obs_node);

#endif
