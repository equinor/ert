#ifndef __FIELD_CONFIG_H__
#define __FIELD_CONFIG_H__
#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <ecl_kw.h>


typedef enum { unknown_file = 0 , rms_roff_file = 1 , ecl_kw_file = 2 , ecl_grdecl_file = 3} field_file_type;
typedef struct field_config_struct field_config_type;

struct field_config_struct {
  CONFIG_STD_FIELDS;
  int nx,ny,nz;
  int sx,sy,sz;
  int logmode;
  const int *index_map;
  
  void * min_value;
  void * max_value;
  int           sizeof_ctype;
  ecl_type_enum ecl_type;
  bool fmt_file;
  bool endian_swap;
  bool limits_set;
  bool write_compressed;
};


bool                field_config_write_compressed(const field_config_type * );
field_file_type     field_config_guess_file_type(const char * , bool);
field_file_type     field_config_manual_file_type(const char * );
ecl_type_enum       field_config_get_ecl_type(const field_config_type * );
void                field_config_get_dims(const field_config_type * , int * , int * , int *);
field_config_type * field_config_alloc(const char * , ecl_type_enum ,  int , int , int , int , const int * , int);
void                field_config_free(field_config_type *);
void                field_config_set_io_options(const field_config_type * , bool *, bool *);
int                 field_config_get_volume(const field_config_type * );
void                field_config_set_ecl_kw_name(field_config_type * , const char * );
void                field_config_set_ecl_type(field_config_type *  , ecl_type_enum );
void                field_config_set_eclfile(field_config_type * , const char * );
void                field_config_set_limits(field_config_type * , void * , void * );
void                field_config_apply_limits(const field_config_type * , void *);
int                 field_config_get_byte_size(const field_config_type * );
int                 field_config_get_active_size(const field_config_type * );
int                 field_config_get_sizeof_ctype(const field_config_type * );
int                 field_config_global_index(const field_config_type * , int , int , int );
void                field_config_get_ijk(const field_config_type * , int , int * , int * , int *);

/*Generated headers */
CONFIG_GET_ECL_KW_NAME_HEADER(field);
VOID_FUNC_HEADER(field_config_free);
#endif
