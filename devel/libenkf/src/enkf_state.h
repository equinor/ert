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
#include <lsf_jobs.h>
#include <sched_file.h>



typedef struct enkf_state_struct enkf_state_type;

void               enkf_state_set_data_kw(enkf_state_type *, const char * , const char * );
void               enkf_state_add_data_kw(enkf_state_type *, const char * , const char * );
void               enkf_state_init_eclipse(enkf_state_type *, const sched_file_type * , int , int);
enkf_fs_type     * enkf_state_get_fs_ref(const enkf_state_type *);
bool 		   enkf_state_get_analyzed(const enkf_state_type * );
void 		   enkf_state_set_analyzed(enkf_state_type * , bool );
void               enkf_state_swapout_node(const enkf_state_type * , const char *);
void               enkf_state_swapin_node(const enkf_state_type *  , const char *);
meas_vector_type * enkf_state_get_meas_vector(const enkf_state_type *);
void              enkf_state_swapout(enkf_state_type * , int , int , state_enum );
void              enkf_state_swapin(enkf_state_type * , int , int , state_enum);
enkf_state_type * enkf_state_copyc(const enkf_state_type * );
void              enkf_state_iset_eclpath(enkf_state_type * , int , const char *);
/*void              enkf_state_add_node(enkf_state_type * , const char * );*/
enkf_state_type * enkf_state_alloc(const enkf_config_type * , int , ecl_store_enum , enkf_fs_type * , const char * , const char * , const char * , meas_vector_type * );
enkf_node_type  * enkf_state_get_node(const enkf_state_type * , const char * );
void              enkf_state_del_node(enkf_state_type * , const char * );
void              enkf_state_load_ecl_summary(enkf_state_type * , bool , int );
void            * enkf_state_load_ecl_summary_void(void * );
void            * enkf_state_load_ecl_restart_void(void * );
void            * enkf_state_load_ecl_void(void * );
void              enkf_state_load_ecl(enkf_state_type * , enkf_obs_type * , bool , int , int);
void              enkf_state_add_lsf_job(enkf_state_type *  , lsf_pool_type * , int , int );
void            * enkf_state_run_eclipse__(void * );


void              enkf_state_add_node(enkf_state_type * , const char *  , const enkf_config_node_type * );
void              enkf_state_load_ecl_restart(enkf_state_type * , bool , int );
void              enkf_state_sample(enkf_state_type * , int);
void              enkf_state_fwrite(const enkf_state_type * , int , int  , state_enum );
void              enkf_state_ens_read(       enkf_state_type * , const char * , int);
void 		  enkf_state_ecl_write(const enkf_state_type * , int , int);
void              enkf_state_ecl_read(enkf_state_type * , const ecl_block_type *);
void              enkf_state_free(enkf_state_type * );
void              enkf_state_apply(enkf_state_type * , enkf_node_ftype1 * , int );
void              enkf_state_serialize(enkf_state_type * , size_t);
void              enkf_state_set_iens(enkf_state_type *  , int );
int               enkf_state_get_iens(const enkf_state_type * );
void 		  enkf_state_set_run_path(enkf_state_type * , const char*);
void 		  enkf_state_set_eclbase(enkf_state_type * , const char*);
void              enkf_state_initialize(enkf_state_type * );

void enkf_ensembleemble_update(enkf_state_type ** , int  , size_t , const double * );
#endif
