#include <stdlib.h>
#include <ecl_kw.h>
#include <field_config.h>
#include <field.h>
#include <ecl_grid.h>
#include <util.h>
#include <string.h>


static bool ENDIAN_FLIP = true;
/*static int * INDEX_MAP   = NULL;*/
static int NX , NY , NZ , ACTIVE_SIZE;
static field_config_type * FIELD_CONFIG; 


void field_inter_static_init__(const char * _egrid_file , const int * egrid_file_len) {
  char * egrid_file = util_alloc_cstring(_egrid_file , egrid_file_len);
  ecl_grid_type * ecl_grid = ecl_grid_alloc_EGRID(egrid_file , ENDIAN_FLIP);
  
  const int * INDEX_MAP   = ecl_grid_alloc_index_map(ecl_grid);
  ecl_grid_get_dims(ecl_grid , &NX , &NY , &NZ , &ACTIVE_SIZE);
  FIELD_CONFIG            = field_config_alloc("KW" , ecl_float_type , NX , NY , NZ , ACTIVE_SIZE , INDEX_MAP , 0);
  
  ecl_grid_free(ecl_grid);
  free(egrid_file);
}



void field_inter_load__(const char *_filename , const int * filename_len , const char *_kw , const int *kw_len , const char * fortran_type_string , void * fortran_data ) {
  char * filename = util_alloc_cstring(_filename , filename_len);
  char * kw       = util_alloc_cstring(_kw       , kw_len);
  int shared_byte_size = ACTIVE_SIZE * 4;
  
  if (strncmp(fortran_type_string , "REAL" , 4) == 0)
    field_config_set_ecl_type(FIELD_CONFIG , ecl_float_type);
  else if (strncmp(fortran_type_string , "DOUB" , 4) == 0) {
    shared_byte_size = ACTIVE_SIZE * 8;
    field_config_set_ecl_type(FIELD_CONFIG , ecl_double_type);
  } else if (strncmp(fortran_type_string , "INTE" , 4) == 0)
    field_config_set_ecl_type(FIELD_CONFIG , ecl_int_type);
  else {
    fprintf(stderr,"%s: Fortran_type_string: %4s not recognized (valid: INTE/REAL/DOUB) - aborting \n",__func__ , fortran_type_string);
    abort();
  }

  field_config_set_ecl_kw_name(FIELD_CONFIG , kw);
  {
    field_type * field = field_alloc_shared(FIELD_CONFIG , fortran_data , shared_byte_size);
    field_fload(field , filename , ENDIAN_FLIP);
    field_free(field);
  }
  
  free(kw);
  free(filename);
}



void field_inter_load3d__(const char *_filename , const int * filename_len , const char *_kw , const int *kw_len , const char * fortran_type_string , void * fortran_data ) {
  char * filename = util_alloc_cstring(_filename , filename_len);
  char * kw       = util_alloc_cstring(_kw       , kw_len);
  int shared_byte_size = ACTIVE_SIZE * 4;
  ecl_type_enum target_type;
  double fill_double = 0.0;
  float  fill_float  = 0.0;
  int    fill_int    = 0;
  void *fill_ptr;

  if (strncmp(fortran_type_string , "REAL" , 4) == 0) {
    field_config_set_ecl_type(FIELD_CONFIG , ecl_float_type);
    target_type = ecl_float_type;
    fill_ptr = &fill_float;
  } else if (strncmp(fortran_type_string , "DOUB" , 4) == 0) {
    shared_byte_size = ACTIVE_SIZE * 8;
    field_config_set_ecl_type(FIELD_CONFIG , ecl_double_type);
    target_type = ecl_double_type;
    fill_ptr = &fill_double;
  } else if (strncmp(fortran_type_string , "INTE" , 4) == 0) {
    field_config_set_ecl_type(FIELD_CONFIG , ecl_int_type);
    target_type = ecl_int_type;
    fill_ptr = &fill_int;
  } else {
    fprintf(stderr,"%s: Fortran_type_string: %4s not recognized (valid: INTE/REAL/DOUB) - aborting \n",__func__ , fortran_type_string);
    abort();
  }

  field_config_set_ecl_kw_name(FIELD_CONFIG , kw);
  {
    field_type * field = field_alloc( FIELD_CONFIG );
    field_fload(field , filename , ENDIAN_FLIP);
    field_export3D(field , fortran_data , false , target_type , fill_ptr);
    field_free(field);
  }
  
  free(kw);
  free(filename);
}
