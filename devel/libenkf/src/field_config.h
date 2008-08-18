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



/*Hmm these two looked suspiciously similar ... */
typedef enum { unknown_file  		= 0,
	       rms_roff_file 		= 1,
	       ecl_kw_file   		= 2,       /* ecl_kw format either packed (i.e. active cells) *or* all cells - used when reading from file. */
	       ecl_kw_file_active_cells = 3,       /* ecl_kw format, only active cells - used writing to file. */
	       ecl_kw_file_all_cells    = 4,       /* ecl_kw_format, all cells - used when writing to file. */
	       ecl_grdecl_file          = 5} field_file_format_type;

	        
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
  int nx,ny,nz;
  int sx,sy,sz;
  int logmode;
  const int *index_map;
  
  void 	      * min_value;
  void        * max_value;
  int           sizeof_ctype;

  field_file_format_type  ecl_export_format;
  ecl_type_enum           ecl_type;
  field_init_type         init_type; 
  char        	* base_file;
  char        	* perturbation_config_file;
  char          * layer_config_file;  
  path_fmt_type * init_file_fmt;

  bool fmt_file;
  bool endian_swap;
  bool limits_set;
  bool write_compressed;
  bool add_perturbation;
};


bool                    field_config_get_endian_swap(const field_config_type * );
bool                    field_config_write_compressed(const field_config_type * );
field_file_format_type  field_config_guess_file_type(const char * , bool);
field_file_format_type  field_config_manual_file_type(const char * );
ecl_type_enum           field_config_get_ecl_type(const field_config_type * );
rms_type_enum           field_config_get_rms_type(const field_config_type * );
void                    field_config_get_dims(const field_config_type * , int * , int * , int *);
field_config_type     * field_config_alloc_dynamic(const char * , int , int , int , int , const int * );
field_config_type     * field_config_alloc_parameter_no_init(const char *, int, int, int, int, const int *);
field_config_type     * field_config_alloc_parameter(const char * , int , int , int , int  , const int * , int , field_init_type  , int  , const char ** );
void                    field_config_free(field_config_type *);
void                    field_config_set_io_options(const field_config_type * , bool *, bool *);
int                     field_config_get_volume(const field_config_type * );
void                    field_config_set_ecl_kw_name(field_config_type * , const char * );
void                    field_config_set_ecl_type(field_config_type *  , ecl_type_enum );
void                    field_config_set_eclfile(field_config_type * , const char * );
void                    field_config_set_limits(field_config_type * , void * , void * );
void                    field_config_apply_limits(const field_config_type * , void *);
int                     field_config_get_byte_size(const field_config_type * );
int                     field_config_get_active_size(const field_config_type * );
int                     field_config_get_sizeof_ctype(const field_config_type * );
int                     field_config_global_index(const field_config_type * , int , int , int );
void                    field_config_get_ijk(const field_config_type * , int , int * , int * , int *);
field_init_type         field_config_get_init_type(const field_config_type * );
char                  * field_config_alloc_init_file(const field_config_type * , int );
field_file_format_type  field_config_get_ecl_export_format(const field_config_type * );


/*Generated headers */
CONFIG_GET_ECL_KW_NAME_HEADER(field);
VOID_FREE_HEADER(field_config);
#endif
