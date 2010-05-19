#ifndef __ENKF_CONFIG_NODE_H__
#define __ENKF_CONFIG_NODE_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stringlist.h>
#include <enkf_types.h>
#include <enkf_macros.h>
#include <ecl_grid.h>
#include <field_trans.h>

typedef void   (config_free_ftype)   (void *);
typedef int    (get_data_size_ftype) (const void *);


typedef struct enkf_config_node_struct enkf_config_node_type;
typedef struct enkf_node_struct        enkf_node_type;



  enkf_config_node_type * enkf_config_node_alloc_gen_kw( const char * key              , 
                                                         const char * enkf_outfile_fmt ,   /* The include file created by ERT for the forward model. */
                                                         const char * template_file    , 
                                                         const char * parameter_file   ,
                                                         const char * min_std_file     ,
                                                         const char * init_file_fmt );


  void enkf_config_node_update_gen_kw( enkf_config_node_type * config_node ,
                                       const char * enkf_outfile_fmt ,   /* The include file created by ERT for the forward model. */
                                       const char * template_file    , 
                                       const char * parameter_file   ,
                                       const char * min_std_file     ,
                                       const char * init_file_fmt );
  

enkf_config_node_type * enkf_config_node_alloc(enkf_var_type         ,
					       enkf_impl_type        ,
					       const char          * ,
					       const char          * , 
					       const char          * , 
					       const void          * );


enkf_config_node_type * enkf_config_node_alloc_summary( const char * key );


void enkf_config_node_update_state_field( enkf_config_node_type * config_node , int truncation , double value_min , double value_max );

enkf_config_node_type * enkf_config_node_alloc_state_field( const char * key              ,
                                                            ecl_grid_type * ecl_grid      , 
                                                            int truncation                ,
                                                            double value_min              , 
                                                            double value_max              ,
                                                            field_trans_table_type * trans_table );


void enkf_config_node_update_parameter_field( enkf_config_node_type * config_node , 
                                              const char * enkf_outfile_fmt , 
                                              const char * init_file_fmt , 
                                              const char * min_std_file , 
                                              int truncation , double value_min , double value_max ,
                                              const char * init_transform , 
                                              const char * output_transform );


enkf_config_node_type * enkf_config_node_alloc_parameter_field( const char * key                     ,
                                                                ecl_grid_type * ecl_grid             , 
                                                                const char * enkf_outfile_fmt        , 
                                                                const char * init_file_fmt           , 
                                                                const char * min_std_file            , 
                                                                int truncation                       ,
                                                                double value_min                     , 
                                                                double value_max                     ,            
                                                                field_trans_table_type * trans_table ,
                                                                const char * init_transform          ,
                                                                const char * output_transform );





void enkf_config_node_update_general_field( enkf_config_node_type * config_node , 
                                            const char * enkf_outfile_fmt        , 
                                            const char * enkf_infile_fmt         , 
                                            const char * init_file_fmt           , 
                                            const char * min_std_file            , 
                                            int truncation                       ,
                                            double value_min                     , 
                                            double value_max                     ,            
                                            const char * init_transform          ,
                                            const char * input_transform         ,
                                            const char * output_transform );


enkf_config_node_type * enkf_config_node_alloc_general_field( const char * key                     ,
                                                              ecl_grid_type * ecl_grid             , 
                                                              const char * enkf_outfile_fmt        , 
                                                              const char * enkf_infile_fmt         , 
                                                              const char * init_file_fmt           , 
                                                              const char * min_std_file            , 
                                                              int truncation                       ,
                                                              double value_min                     , 
                                                              double value_max                     ,            
                                                              field_trans_table_type * trans_table ,
                                                              const char * init_transform          ,
                                                              const char * input_transform         ,
                                                              const char * output_transform );



/*****************************************************************/

enkf_config_node_type * enkf_config_node_new_gen_kw( const char * key );
enkf_config_node_type * enkf_config_node_new_field( const char * key , ecl_grid_type * ecl_grid, field_trans_table_type * trans_table);
bool                    enkf_config_node_is_valid( const enkf_config_node_type * config_node );
int                     enkf_config_node_get_data_size( const enkf_config_node_type * node , int report_step);					
char                  * enkf_config_node_alloc_infile(const enkf_config_node_type * , int );
char                  * enkf_config_node_alloc_outfile(const enkf_config_node_type * , int );
int                     enkf_config_node_get_num_obs( const enkf_config_node_type * config_node );
int                     enkf_config_node_load_obs( const enkf_config_node_type * config_node , const enkf_obs_type * enkf_obs ,const char * key_index , int obs_count , time_t * sim_time , double * y , double * std);
const stringlist_type * enkf_config_node_get_obs_keys(const enkf_config_node_type *);
void              	enkf_config_node_add_obs_key(enkf_config_node_type *  , const char * );
void                    enkf_config_node_clear_obs_keys(enkf_config_node_type * config_node);
void 		  	enkf_config_node_free(enkf_config_node_type * );
bool              	enkf_config_node_include_type(const enkf_config_node_type * , int );
int  		  	enkf_config_node_get_serial_size(enkf_config_node_type *, int *);
bool 		  	enkf_config_node_include_type(const enkf_config_node_type * , int);
enkf_impl_type    	enkf_config_node_get_impl_type(const enkf_config_node_type *);
enkf_var_type     	enkf_config_node_get_var_type(const enkf_config_node_type *);
      void     *  	enkf_config_node_get_ref(const enkf_config_node_type * );
const char     *  	enkf_config_node_get_key(const enkf_config_node_type * );
void                    enkf_config_node_init_internalization(enkf_config_node_type * );
void                    enkf_config_node_set_min_std( enkf_config_node_type * config_node , enkf_node_type * min_std );
const            char * enkf_config_node_get_min_std_file( const enkf_config_node_type * config_node );
const            char * enkf_config_node_get_enkf_outfile( const enkf_config_node_type * conifg_node );

void enkf_config_node_set_internalize(enkf_config_node_type * node, int report_step);
bool enkf_config_node_internalize(const enkf_config_node_type * node, int report_step);

/*
  The enkf_node_free() function declaration is in the enkf_config_node.h header,
  because the enkf_config_node needs to know how to free the min_std node.
*/
void                  enkf_node_free(enkf_node_type *enkf_node);
const enkf_node_type * enkf_config_node_get_min_std( const enkf_config_node_type * config_node );

UTIL_SAFE_CAST_HEADER(enkf_config_node);
VOID_FREE_HEADER(enkf_config_node);
#ifdef __cplusplus
}
#endif
#endif
