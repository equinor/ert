#ifndef __ENKF_STATE_H__
#define __ENKF_STATE_H__


#include <fortio.h>
#include <stdbool.h>
#include <enkf_config.h>
#include <enkf_types.h>
#include <enkf_node.h>
#include <enkf_util.h>
#include <enkf_obs.h>
#include <ecl_block.h>
#include <meas_vector.h>
#include <enkf_fs.h>
#include <sched_file.h>
#include <ext_joblist.h>
#include <stringlist.h>
#include <job_queue.h>

typedef struct enkf_state_struct    enkf_state_type;
typedef struct OLD_enkf_run_info_struct enkf_run_info_type;

int                enkf_state_get_report_step(const enkf_state_type * );
void               enkf_state_measure( const enkf_state_type *  , enkf_obs_type * );
void               enkf_state_set_state(enkf_state_type * , int , state_enum );
void               enkf_state_set_data_kw(enkf_state_type *, const char * , const char * );
void               enkf_state_init_eclipse(enkf_state_type *, const sched_file_type * , int , state_enum , int, int, const stringlist_type * );
enkf_fs_type     * enkf_state_get_fs_ref(const enkf_state_type *);
bool               enkf_state_get_analyzed(const enkf_state_type * );
void               enkf_state_set_analyzed(enkf_state_type * , bool );
void               enkf_state_swapout_node(const enkf_state_type * , const char *);
void               enkf_state_swapin_node(const enkf_state_type *  , const char *);
meas_vector_type * enkf_state_get_meas_vector(const enkf_state_type *);
void               enkf_state_swapout(enkf_state_type * , int);
void               enkf_state_swapin(enkf_state_type * , int);
enkf_state_type  * enkf_state_copyc(const enkf_state_type * );
void               enkf_state_iset_eclpath(enkf_state_type * , int , const char *);
enkf_state_type  * enkf_state_alloc(const enkf_config_type * , int , ecl_store_enum , enkf_fs_type * , ext_joblist_type * , job_queue_type * , sched_file_type * , path_fmt_type * , path_fmt_type * , path_fmt_type * ,  meas_vector_type * , enkf_obs_type *);
enkf_node_type   * enkf_state_get_node(const enkf_state_type * , const char * );
void               enkf_state_del_node(enkf_state_type * , const char * );
void               enkf_state_load_ecl_summary(enkf_state_type * , bool , int );
void             * enkf_state_ecl_load__(void * );
void               enkf_state_ecl_load(enkf_state_type * , enkf_obs_type * , bool , int , int);
void             * enkf_state_run_eclipse__(void * );
void             * enkf_state_start_eclipse__(void * );
void             * enkf_state_complete_eclipse__(void * );


void               enkf_state_add_node(enkf_state_type * , const char *  , const enkf_config_node_type * );
void               enkf_state_load_ecl_restart(enkf_state_type * , bool , int );
void               enkf_state_sample(enkf_state_type * , int);
void               enkf_state_fwrite_as(enkf_state_type *  , int  , int  , state_enum );
void               enkf_state_fwrite(const enkf_state_type * , int );
void               enkf_state_ens_read(       enkf_state_type * , const char * , int);
void               enkf_state_ecl_write(enkf_state_type * , int);
void               enkf_state_ecl_read(enkf_state_type * , const ecl_block_type *);
void               enkf_state_free(enkf_state_type * );
void               enkf_state_apply(enkf_state_type * , enkf_node_ftype1 * , int );
void               enkf_state_serialize(enkf_state_type * , size_t);
void               enkf_state_set_iens(enkf_state_type *  , int );
int                enkf_state_get_iens(const enkf_state_type * );
const char       * enkf_state_get_run_path(const enkf_state_type * );
void               enkf_state_set_run_path(enkf_state_type * , const char*);
void               enkf_state_set_eclbase(enkf_state_type * , const char*);
void               enkf_state_initialize(enkf_state_type * );

void enkf_ensemble_update(enkf_state_type ** , int  , size_t , const double * );

/*****************************************************************/
void enkf_state_set_run_parameters(enkf_state_type * state , int init_step , state_enum init_state , int step1 , int step2 , bool load_results , bool unlink_run_path , const stringlist_type * forward_model);
bool enkf_state_run_OK(const enkf_state_type * );


enkf_run_info_type * enkf_run_info_alloc(enkf_state_type * ,
					 job_queue_type  * ,
					 enkf_obs_type   * ,
					 sched_file_type * ,
                                         bool              ,
					 int               ,
					 state_enum        ,
					 int               ,
					 int               ,
					 bool              ,
					 bool              ,
					 stringlist_type * );
bool enkf_run_info_OK(const enkf_run_info_type * );
void enkf_run_info_free(enkf_run_info_type * );
#endif
