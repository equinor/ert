#ifndef __GEN_DATA_CONFIG_H__
#define __GEN_DATA_CONFIG_H__
#include <enkf_macros.h>
#include <stdbool.h>
#include <gen_data_active.h>

typedef enum { gen_data_undefined = 0,  
	       ASCII           	  = 1,   /* The file is ASCII file with a vector of numbers formatted with "%g". */
	       ASCII_template  	  = 2,   /* The data is inserted into a user defined template file.*/
	       binary_double   	  = 3,   /* The data is in a binary file with doubles. */
	       binary_float    	  = 4}   /* Well ... */
               gen_data_format_type;
	       

/* 
   Observe that the format ASCII_template can *NOT* be used for
   loading files.
*/

typedef struct gen_data_config_struct gen_data_config_type;

gen_data_format_type    gen_data_config_get_input_format ( const gen_data_config_type * );
gen_data_format_type    gen_data_config_get_output_format ( const gen_data_config_type * );
char                  * gen_data_config_alloc_initfile(const gen_data_config_type *  , int );
ecl_type_enum           gen_data_config_get_internal_type(const gen_data_config_type * );
int                     gen_data_config_get_byte_size(const gen_data_config_type * );
int                     gen_data_config_get_data_size(const gen_data_config_type * );
gen_data_config_type  * gen_data_config_alloc(gen_data_format_type , gen_data_format_type , const char * );
gen_data_config_type  * gen_data_config_alloc_with_template(gen_data_format_type , const char * , const char * , const char *);
void             	gen_data_config_free(gen_data_config_type * );
void             	gen_data_config_assert_size(gen_data_config_type *  , int);
const bool     * 	gen_data_config_get_iactive(const gen_data_config_type * );
void                    gen_data_config_ecl_write(const gen_data_config_type *  , const char * , char * );
void                    gen_data_config_get_template_data( const gen_data_config_type * , char ** , int * , int * , int *);
gen_data_config_type  * gen_data_config_fscanf_alloc(const char * );

GET_ACTIVE_LIST_HEADER(gen_data)
VOID_FREE_HEADER(gen_data_config)
VOID_CONFIG_ACTIVATE_HEADER(gen_data);

#endif
