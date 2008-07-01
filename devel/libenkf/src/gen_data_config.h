#ifndef __GEN_DATA_CONFIG_H__
#define __GEN_DATA_CONFIG_H__

typedef struct gen_data_config_struct gen_data_config_type; 


void    	        gen_data_config_get_ecl_file(const gen_data_config_type * , int , char ** , char ** );
void    	        gen_data_config_free(gen_data_config_type * );
gen_data_config_type  * gen_data_config_fscanf_alloc(const char * );
bool                    gen_data_config_is_active(const gen_data_config_type *  , int );
void                    gen_data_config_assert_metadata(gen_data_config_type * , int  , int , ecl_type_enum , const char * );

#endif
