#ifndef __FIELD_CONFIG_H__
#define __FIELD_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <ecl_kw.h>
#include <path_fmt.h>
#include <rms_file.h>
#include <ecl_grid.h>
#include <active_list.h>
#include <field_active.h>
#include <field_trans.h>
#include <stringlist.h>
#include <field_common.h>



/**
   This is purely a convenience structure used during initialization,
   to denote which arguments are required and, which should be
   defualted.

*/
    
typedef enum {
  ECLIPSE_RESTART   = 1,
  ECLIPSE_PARAMETER = 2,
  GENERAL           = 3
} field_type_enum;
 

/** 
    The field_file_format_type denotes different ways to store a
    field. Unfortunately the different elements in the enum definition
    have somewhat different properties:


    1. ecl_kw_file is for input - either pack or unpacked.

    2. ecl_kw_file_active_cells / ecl_kw_file_all_cells are for output.

    3. Except for ecl_restart_file all formats are for A FILE (with a
       filename), more or less assuming that this field is the only
       content in the file, whereas ecl_restart_file is for a restart
       block, and not a file.

    This has some slightly unlogical consequences:

     1. The enum has 'file_format' in the name, but ecl_restart_file
        is not a file.

     2. The functions which guess/determine a file type can not return
        all possible values of the enum.

     3. Treatment is not symmetric for input/output.

*/
    



typedef enum { UNDEFINED_FORMAT         = 0,
	       RMS_ROFF_FILE 		= 1,
	       ECL_KW_FILE   		= 2,       /* ecl_kw format either packed (i.e. active cells) *or* all cells - used when reading from file. */
	       ECL_KW_FILE_ACTIVE_CELLS = 3,       /* ecl_kw format, only active cells - used writing to file. */
	       ECL_KW_FILE_ALL_CELLS    = 4,       /* ecl_kw_format, all cells - used when writing to file. */
	       ECL_GRDECL_FILE          = 5, 
               ECL_FILE                 = 6,       /* Assumes packed on export. */
               FILE_FORMAT_NULL         = 7}       /* Used when the guess functions are given NULL to check -should never be read. */ field_file_format_type; 

	        
/* active_cells currently not really implemented */



  

void                    field_config_update_state_field( field_config_type * config, int truncation, double min_value , double max_value);


void field_config_update_parameter_field( field_config_type * config , int truncation, double min_value , double max_value, 
                                          field_file_format_type export_format , 
                                          const char * init_file_fmt, 
                                          const char * init_transform , const char * output_transform );


void field_config_update_general_field( field_config_type * config , int truncation, double min_value , double max_value, 
                                        field_file_format_type export_format , /* This can be guessed with the field_config_default_export_format( ecl_file ) function. */
                                        const char * init_file_fmt, 
                                        const char * init_transform , 
                                        const char * input_transform , 
                                        const char * output_transform );


field_config_type * field_config_alloc_empty( const char * ecl_kw_name , ecl_grid_type * ecl_grid , field_trans_table_type * trans_table );



void                    field_config_get_ijk( const field_config_type * config , int active_index , int *i , int * j , int * k);
field_type            * field_config_get_min_std( const field_config_type * field_config );
const char            * field_config_default_extension(field_file_format_type , bool );
bool                    field_config_write_compressed(const field_config_type * );
field_file_format_type  field_config_guess_file_type(const char * );
field_file_format_type  field_config_manual_file_type(const char * , bool);
ecl_type_enum           field_config_get_ecl_type(const field_config_type * );
rms_type_enum           field_config_get_rms_type(const field_config_type * );
void                    field_config_get_dims(const field_config_type * , int * , int * , int *);
int                     field_config_get_nx(const field_config_type * config );
int                     field_config_get_ny(const field_config_type * config );
int                     field_config_get_nz(const field_config_type * config );
void                    field_config_free(field_config_type *);
int                     field_config_get_volume(const field_config_type * );
void                    field_config_set_ecl_kw_name(field_config_type * , const char * );
void                    field_config_set_ecl_type(field_config_type *  , ecl_type_enum );
void                    field_config_set_eclfile(field_config_type * , const char * );
const bool            * field_config_get_iactive(const field_config_type * );
int                     field_config_get_byte_size(const field_config_type * );
int                     field_config_get_sizeof_ctype(const field_config_type * );
int                     field_config_active_index(const field_config_type * , int , int , int );
void                    field_config_get_ijk(const field_config_type * , int , int * , int * , int *);
bool                    field_config_ijk_valid(const field_config_type *  , int  , int  , int );
bool                    field_config_ijk_active(const field_config_type * config , int i , int j , int k);
bool                    field_config_active_cell(const field_config_type *  , int , int , int);
bool                    field_config_enkf_init(const field_config_type * );
char                  * field_config_alloc_init_file(const field_config_type * , int );
field_file_format_type  field_config_get_export_format(const field_config_type * );
field_file_format_type  field_config_get_import_format(const field_config_type * );
void                    field_config_set_all_active(field_config_type * );
void                    field_config_set_key(field_config_type * , const char *);
void                    field_config_enkf_OFF(field_config_type * );
bool                    field_config_enkf_mode(const field_config_type * config);
void                    field_config_scanf_ijk(const field_config_type *  , bool , const char * , int , int * , int * , int * , int *);
const char            * field_config_get_key(const field_config_type * );
field_func_type       * field_config_get_init_transform(const field_config_type * );
field_func_type       * field_config_get_output_transform(const field_config_type * );
field_func_type       * field_config_get_input_transform(const field_config_type * );
  //void                    field_config_set_output_transform(field_config_type * config , field_func_type * );
void                    field_config_assert_binary( const field_config_type *  , const field_config_type *  , const char * );
void                    field_config_assert_unary( const field_config_type *  , const char * );
void                    field_config_activate(field_config_type *  , active_mode_type  , void * );

const char            * field_config_get_init_transform_name( const field_config_type * field_config );
const char            * field_config_get_input_transform_name( const field_config_type * field_config );
const char            * field_config_get_output_transform_name( const field_config_type * field_config );

void            	field_config_set_truncation_from_strings(field_config_type * , const char * , const char **);
void            	field_config_set_truncation(field_config_type * , truncation_type , double , double );
truncation_type 	field_config_get_truncation(const field_config_type * , double * , double *);
const ecl_grid_type   * field_config_get_grid(const field_config_type * );
const char            * field_config_get_grid_name( const field_config_type * );

int                     field_config_parse_user_key(const field_config_type * config, const char * index_key , int *_i , int *_j , int *_k);

field_file_format_type    field_config_default_export_format(const char * filename);
const char              * field_config_get_input_transform_name( const field_config_type * field_config ) ;
const char              * field_config_get_output_transform_name( const field_config_type * field_config ) ;
const char              * field_config_get_init_transform_name( const field_config_type * field_config ) ;


/*Generated headers */
SAFE_CAST_HEADER(field_config);
CONFIG_GET_ECL_KW_NAME_HEADER(field);
VOID_FREE_HEADER(field_config);
GET_ACTIVE_LIST_HEADER(field);
VOID_GET_DATA_SIZE_HEADER(field);
#ifdef __cplusplus
}
#endif
#endif
