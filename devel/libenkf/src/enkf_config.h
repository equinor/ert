#ifndef __ENKF_CONFIG_H__
#define __ENKF_CONFIG_H__
#include <enkf_config_node.h>
#include <stdbool.h>

typedef struct enkf_config_struct enkf_config_type;

enkf_impl_type      enkf_config_impl_type(const enkf_config_type *, const char * );
bool                enkf_config_get_endian_swap(const enkf_config_type * );
bool                enkf_config_get_fmt_file(const enkf_config_type * );
const char        * enkf_config_get_data_file(const enkf_config_type * );
bool                enkf_config_has_key(const enkf_config_type * , const char *);
enkf_config_type  * enkf_config_alloc(int ens_size            , 
				      const char * data_file  , 
				      const char * _run_path  , 
				      const char * _eclbase   , 
				      bool fmt_file 	     ,
				      bool unified  	     ,         
				      bool endian_swap);
const char       ** enkf_config_get_well_list_ref(const enkf_config_type * , int *);
void                enkf_config_add_well(enkf_config_type * , const char * , int , const char ** );
void                enkf_config_add_gen_kw(enkf_config_type * , const char * );
void                enkf_config_add_type(enkf_config_type * , const char * , enkf_var_type , enkf_impl_type , const char * , const void *);

const enkf_config_node_type * enkf_config_get_config_ref(const enkf_config_type * , const char * );
void                  	      enkf_config_add_type0(enkf_config_type * , const char * , int , enkf_var_type , enkf_impl_type );

char 			    * enkf_config_alloc_run_path(const enkf_config_type * , int );
char 			    * enkf_config_alloc_eclbase(const enkf_config_type  * , int );
int                           enkf_config_get_ens_size(const enkf_config_type * );
void enkf_config_free(enkf_config_type * );
bool enkf_config_get_unified(const enkf_config_type * );

#endif
