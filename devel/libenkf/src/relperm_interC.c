#include <relperm.h>
#include <relperm_config.h>
#include <util.h>

static relperm_config_type * RELPERM_CONFIG = NULL;
static relperm_type ** RELPERM_LIST = NULL;


/*****************************************************************/

/*
All functions in relperm_interC.c are called from fortran _enkf_inter.F90, and therefore no relperm_interC.h exists 
Notation __ at the end of the function name indicate that the function is internal 
*/


void relperm_inter_init__(const char * _config_file, const int * config_file_len, const char * _table_file, const int * table_file_len, const int * ens_size, const int * n_relperm) {
  int iens;
  char * config_file = util_alloc_cstring(_config_file, config_file_len);
  char * table_file  = util_alloc_cstring(_table_file, table_file_len);
  RELPERM_CONFIG = relperm_config_fscanf_alloc(config_file,table_file);
  
  if(*n_relperm != relperm_config_get_data_size(RELPERM_CONFIG)){
    fprintf(stderr, "%s: size mismatch config_file:%d mod_dimension.F90/relpdims:%d -aborting \n",__func__,relperm_config_get_data_size(RELPERM_CONFIG),*n_relperm);
    /*
    abort();
    */
  }
  RELPERM_LIST = malloc(*ens_size * sizeof * RELPERM_LIST);
  
  for (iens = 0;iens < *ens_size; iens++){
    RELPERM_LIST[iens] = relperm_alloc(RELPERM_CONFIG);
  }
  
  free(config_file);
  free(table_file);

}

void relperm_inter_sample__(const int * iens, double * data){
  relperm_initialize(RELPERM_LIST[(*iens)-1] , 0);
  relperm_get_data(RELPERM_LIST[(*iens)-1],data); 
}

void relperm_inter_ecl_write__(const char * _path, const int * path_len, const double * data, const int * iens){
  char * path = util_alloc_cstring(_path,path_len);
  relperm_ecl_write_f90test(RELPERM_LIST[(*iens)-1],data,path);
  
  /*
  char * file = util_alloc_full_path(path,"RELPERM.INC");
  */

  /*  
  relperm_set_data(RELPERM_LIST[(*iens)-1],data);
  relperm_output_transform(RELPERM_LIST[(*iens)-1]);
  relperm_make_tab(RELPERM_LIST[(*iens)-1]); 
  relperm_ecl_write(RELPERM_LIST[(*iens)-1],file);
  */

  free(path);

}

void relperm_inter_transform_relperm_data__(const int * iens, const double * input_data,double * output_data){
  relperm_set_data(RELPERM_LIST[(*iens)-1], input_data);
  relperm_output_transform(RELPERM_LIST[(*iens)-1]);
  relperm_get_output_data(RELPERM_LIST[(*iens)-1],output_data);
}

void relperm_inter_get_relperm_description__(const int * relperm_nr, char * _description, const int * description_len){
  /*  
      const char * description = relperm_get_name(RELPERM_LIST[0],(*relperm_nr)-1);
      util_memcpy_string_C2f90(description, _description, *description_len);
  */
  printf("relperm_inter_get_descprition not implemented");
}
