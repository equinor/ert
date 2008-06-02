#ifndef __ENKF_CONFIG_H__
#define __ENKF_CONFIG_H__
#include <enkf_config_node.h>
#include <stdbool.h>
#include <time.h>
#include <sched_file.h>
#include <enkf_types.h>
#include <enkf_site_config.h>
#include <ecl_queue.h>
#include <ext_joblist.h>

typedef struct enkf_config_struct enkf_config_type;

void        * enkf_config_get_data_kw(const enkf_config_type *  , const char * );
char       ** enkf_config_alloc_data_kw_key_list(const enkf_config_type * , int * );

enkf_impl_type      enkf_config_impl_type(const enkf_config_type *, const char * );
bool                enkf_config_get_endian_swap(const enkf_config_type * );
bool                enkf_config_get_fmt_file(const enkf_config_type * );
const char        * enkf_config_get_data_file(const enkf_config_type * );
bool                enkf_config_has_key(const enkf_config_type * , const char *);
enkf_config_type  * enkf_config_alloc(int ens_size            , 
				      const int  *  , 
				      const char *  , 
				      const char * data_file  , 
				      const char * _run_path  , 
				      const char * _eclbase   , 
				      bool fmt_file 	     ,
				      bool unified  	     ,         
				      bool endian_swap);
const char       ** enkf_config_get_forward_model(const enkf_config_type * , int * );
void                enkf_config_add_well(enkf_config_type * , const char * , int , const char ** );
void                enkf_config_add_gen_kw(enkf_config_type * , const char * );
void                enkf_config_add_type(enkf_config_type * , const char * , enkf_var_type , enkf_impl_type , const char * , const void *);
time_t              enkf_config_get_start_date(const enkf_config_type * );
const        char * enkf_config_get_obs_config_file(const enkf_config_type * );
const        char * enkf_config_get_ens_path(const enkf_config_type * config);
char             ** enkf_config_alloc_keylist(const enkf_config_type * , int *);

const enkf_config_node_type * enkf_config_get_node_ref(const enkf_config_type * , const char * );
void                          enkf_config_get_grid_dims(const enkf_config_type * , int *, int *, int *, int *);
char 			    * enkf_config_alloc_run_path(const enkf_config_type * , int );
char 			    * enkf_config_alloc_eclbase(const enkf_config_type  * , int );
char                        * enkf_config_alloc_result_path(const enkf_config_type * config , int );
char 			    * enkf_config_alloc_ecl_store_path(const enkf_config_type  * , int );
int                           enkf_config_get_ens_size(const enkf_config_type * );
void enkf_config_free(enkf_config_type * );
bool enkf_config_get_unified(const enkf_config_type * );

enkf_config_type * enkf_config_fscanf_alloc(const char *  ,  enkf_site_config_type * , ext_joblist_type * , int , bool  ,  bool   , bool );
int                enkf_config_get_ens_offset(const enkf_config_type * );
ecl_store_enum     enkf_config_iget_ecl_store(const enkf_config_type * , int );
ecl_queue_type   * enkf_config_alloc_ecl_queue(const enkf_config_type * , const enkf_site_config_type * );
const char       * enkf_config_get_schedule_target_file(const enkf_config_type * );
const char       * enkf_config_get_schedule_src_file(const enkf_config_type * );
const char       * enkf_config_get_init_file(const enkf_config_type * );
#endif
