#ifndef __FIELD_CONFIG_H__
#define __FIELD_CONFIG_H__
#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <ecl_kw.h>
#include <path_fmt.h>
#include <rms_file.h>
#include <ecl_grid.h>




/** 
    The field_file_format_type denotes different ways to store a
    field. Unfortunately the different elements in the enum definition
    have somewhat different properties:


    1. ecl_kw_file is for input - either pack or unpacked.

    2. ecl_kw_file_active_cells / ecl_kw_file_all_cells are for output.

    3. Except for ecl_restart_block all formats are for A FILE (with a
       filename), more or less assuming that this field is the only
       content in the file, whereas ecl_restart_block is for a restart
       block, and not a file.

    This has some slightly unlogical consequences:

     1. The enum has 'file_format' in the name, but ecl_restart_block
        is not a file.

     2. The functions which guess/determine a file type can not return
        all possible values of the enum.

     3. Treatment is not symmetric for input/output.

*/
    



typedef enum { undefined_format         = 0,
	       rms_roff_file 		= 1,
	       ecl_kw_file   		= 2,       /* ecl_kw format either packed (i.e. active cells) *or* all cells - used when reading from file. */
	       ecl_kw_file_active_cells = 3,       /* ecl_kw format, only active cells - used writing to file. */
	       ecl_kw_file_all_cells    = 4,       /* ecl_kw_format, all cells - used when writing to file. */
	       ecl_grdecl_file          = 5, 
               ecl_restart_block        = 6} field_file_format_type;

	        
/* active_cells currently not really implemented */



/* Must be power of two series */
typedef enum { none                   = 0  , /* For restart fields */
               load_unique            = 1  , /* path_fmt_type to load from */ 
               load_base_case         = 2  , /* Filename to load */
	       layer_trends           = 4  , /* Config file */   
	       gaussian_perturbations = 8  , /* Config file */   
} field_init_type;

/*
  gaussian_perturbations : Whether gaussian perturbations should be added.
  layer_trends           : Whether the base case should be baased on layer trends.
  load_base_case         : Base case loaded from file
  load_uniqe             : Members loaded from separate files.
*/
  


typedef struct field_config_struct field_config_type;

struct field_config_struct {
  CONFIG_STD_FIELDS;
  int nx,ny,nz;                       /* The number of elements in the three directions. */ 
  int sx,sy,sz;                       /* The stride in the various directions, i.e. when adressed as one long vector in memory you jump sz elements to iterate along the z direction. */ 
  int logmode;
  const int *index_map;

  bool        * enkf_active;          /* Whether a certain cell is active or not - EnKF wise.*/
  bool          enkf_all_active;      /* Performance gain when all cells are active. */
  void 	      * min_value;
  void        * max_value;
  int           sizeof_ctype;

  field_file_format_type  export_format;
  field_file_format_type  import_format;    
  ecl_type_enum           internal_ecl_type;
  ecl_type_enum           export_ecl_type;
  field_init_type         init_type; 
  char          	* base_file;
  char          	* perturbation_config_file;
  char                  * layer_config_file;  
  path_fmt_type         * init_file_fmt;

  bool __enkf_mode;  /* See doc of functions field_config_set_key() / field_config_enkf_OFF() */
  bool fmt_file;
  bool endian_swap;
  bool limits_set;
  bool write_compressed;
  bool add_perturbation;
};


const char            * field_config_default_extension(field_file_format_type , bool );
bool                    field_config_get_endian_swap(const field_config_type * );
bool                    field_config_write_compressed(const field_config_type * );
field_file_format_type  field_config_guess_file_type(const char * , bool);
field_file_format_type  field_config_manual_file_type(const char * , bool);
ecl_type_enum           field_config_get_ecl_type(const field_config_type * );
rms_type_enum           field_config_get_rms_type(const field_config_type * );
void                    field_config_get_dims(const field_config_type * , int * , int * , int *);
field_config_type     * field_config_alloc_dynamic(const char * , const ecl_grid_type *);
field_config_type     * field_config_alloc_parameter_no_init(const char *, const ecl_grid_type *);
field_config_type     * field_config_alloc_parameter(const char * , const ecl_grid_type * , int , field_init_type  , int  , const char ** );
void                    field_config_free(field_config_type *);
void                    field_config_set_io_options(const field_config_type * , bool *, bool *);
int                     field_config_get_volume(const field_config_type * );
void                    field_config_set_ecl_kw_name(field_config_type * , const char * );
void                    field_config_set_ecl_type(field_config_type *  , ecl_type_enum );
void                    field_config_set_eclfile(field_config_type * , const char * );
void                    field_config_set_limits(field_config_type * , void * , void * );
void                    field_config_apply_limits(const field_config_type * , void *);
const bool            * field_config_get_iactive(const field_config_type * );
int                     field_config_get_byte_size(const field_config_type * );
int                     field_config_get_active_size(const field_config_type * );
int                     field_config_get_sizeof_ctype(const field_config_type * );
int                     field_config_global_index(const field_config_type * , int , int , int );
void                    field_config_get_ijk(const field_config_type * , int , int * , int * , int *);
bool                    field_config_active_cell(const field_config_type *  , int , int , int);
field_init_type         field_config_get_init_type(const field_config_type * );
char                  * field_config_alloc_init_file(const field_config_type * , int );
field_file_format_type  field_config_get_export_format(const field_config_type * );
field_file_format_type  field_config_get_import_format(const field_config_type * );
void                    field_config_set_iactive(field_config_type * , int  , const int *  , const int * , const int *);
void                    field_config_set_all_active(field_config_type * );
void                    field_config_set_key(field_config_type * , const char *);
void                    field_config_enkf_OFF(field_config_type * );
bool                    field_config_enkf_mode(const field_config_type * config);

/*Generated headers */
CONFIG_GET_ECL_KW_NAME_HEADER(field);
VOID_FREE_HEADER(field_config);
#endif
