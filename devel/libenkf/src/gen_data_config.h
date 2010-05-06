#ifndef __GEN_DATA_CONFIG_H__
#define __GEN_DATA_CONFIG_H__
#ifdef __cplusplus 
extern "C" {
#endif

#include <enkf_macros.h>
#include <stdbool.h>
#include <gen_data_active.h>
#include <stringlist.h>
#include <gen_data_common.h>

typedef enum { GEN_DATA_UNDEFINED = 0,  
	       ASCII           	  = 1,   /*   The file is ASCII file with a vector of numbers formatted with "%g".       */
	       ASCII_TEMPLATE  	  = 2,   /*   The data is inserted into a user defined template file.    		 */
	       BINARY_DOUBLE   	  = 3,   /*   The data is in a binary file with doubles. 		       		 */
	       BINARY_FLOAT    	  = 4}   /*   The data is in a binary file with floats.  		       		 */          

gen_data_file_format_type;



/* 
   Observe that the format ASCII_template can *NOT* be used for
   loading files.
*/

char                       * gen_data_config_pop_enkf_infile( gen_data_config_type * config );
char                       * gen_data_config_pop_enkf_outfile( gen_data_config_type * config );
gen_data_type              * gen_data_config_get_min_std( const gen_data_config_type * config );
gen_data_file_format_type    gen_data_config_get_input_format ( const gen_data_config_type * );
gen_data_file_format_type    gen_data_config_get_output_format ( const gen_data_config_type * );
char                  	   * gen_data_config_alloc_initfile(const gen_data_config_type *  , int );
ecl_type_enum         	     gen_data_config_get_internal_type(const gen_data_config_type * );
gen_data_config_type  	   * gen_data_config_alloc_with_options(const char * key , bool , const stringlist_type *);
void                         gen_data_config_free(gen_data_config_type * );
void                         gen_data_config_assert_size(gen_data_config_type *  , int , int);
const bool     *             gen_data_config_get_iactive(const gen_data_config_type * );
void                  	     gen_data_config_ecl_write(const gen_data_config_type *  , const char * , char * );
void                  	     gen_data_config_get_template_data( const gen_data_config_type * , char ** , int * , int * , int *);
gen_data_config_type  	   * gen_data_config_fscanf_alloc(const char * );
const char  *                gen_data_config_get_key( const gen_data_config_type * config);
int                          gen_data_config_get_byte_size( const gen_data_config_type * config , int report_step);
int                          gen_data_config_get_data_size( const gen_data_config_type * config , int report_step);

SAFE_CAST_HEADER(gen_data_config)
GET_ACTIVE_LIST_HEADER(gen_data)
VOID_FREE_HEADER(gen_data_config)

#ifdef __cplusplus
}
#endif
#endif
