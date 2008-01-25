#include <multz.h>
#include <multz_config.h>
#include <util.h>

static multz_config_type  * MULTZ_CONFIG = NULL;
static multz_type        ** MULTZ_LIST        = NULL;

/*****************************************************************/

void multz_inter_init__(const char * _config_file, const int * config_file_len , const int * ens_size, const int * n_multz, const int *nx , const int * ny , const int * nz) {
  int iens;
  char * config_file = util_alloc_cstring(_config_file , config_file_len);
  MULTZ_CONFIG = multz_config_fscanf_alloc(config_file , *nx , *ny , *nz);
  
  if (*n_multz != multz_config_get_data_size(MULTZ_CONFIG)) {
    fprintf(stderr,"%s: size mismatch config_file:%d  mod_dimensions.F90/num_multz:%d - aborting \n",__func__ , multz_config_get_data_size(MULTZ_CONFIG) , *n_multz);
    abort();
  }

  
  MULTZ_LIST = malloc(*ens_size * sizeof * MULTZ_LIST);
  for (iens = 0; iens < *ens_size; iens++) 
    MULTZ_LIST[iens] = multz_alloc(MULTZ_CONFIG);


  free(config_file);
}



/*
  -1 to account for Fortran offset
*/

void multz_inter_ecl_write__(const char *_path , const int * path_len , const double *data , const int *iens) {
  char *path  = util_alloc_cstring(_path , path_len);
  char * file = util_alloc_full_path(path , "MULTZ.INC");

  multz_set_data(MULTZ_LIST[(*iens) - 1]  , data);
  multz_ecl_write(MULTZ_LIST[(*iens) - 1] , file);

  free(file);
  free(path);
}


void multz_inter_ecl_write_avg__(const int * ens_size , const char *_path , const int * path_len) {
  char *path = util_alloc_cstring(_path , path_len);
  
  multz_type *avg_multz = multz_alloc_mean(*ens_size , (const multz_type **) MULTZ_LIST);
  /*
    if (*avg_output == 1)
    multz_direct_ecl_write(avg_multz , path);
    else
  */
  
  multz_ecl_write(avg_multz , path);

  multz_free(avg_multz);
  free(path);
}


void multz_inter_transform_multz_data__(const int * iens , const double * input_data , double * output_data) {
  multz_set_data(MULTZ_LIST[(*iens) - 1] , input_data);
  multz_output_transform(MULTZ_LIST[(*iens) - 1]);
  multz_get_output_data(MULTZ_LIST[(*iens) - 1] , output_data);
}



void multz_get_data__(const int * iens, double * data) {
  multz_get_data(MULTZ_LIST[(*iens) - 1] , data);
}


void multz_inter_sample__(const int * iens, double * data) {
  multz_sample(MULTZ_LIST[(*iens) - 1]);
  multz_get_data(MULTZ_LIST[(*iens) - 1] , data);
}


void multz_inter_truncate__(const int * iens , double * data) {
  multz_set_data(MULTZ_LIST[(*iens) - 1] , data);
  multz_truncate(MULTZ_LIST[(*iens) - 1]);
  multz_get_data(MULTZ_LIST[(*iens) - 1] , data);
}


void multz_inter_get_description__(const int * multz_nr , char * _description , const int * description_len) {
  char * description = multz_alloc_description(MULTZ_LIST[0] , (*multz_nr) - 1);
  util_memcpy_string_C2f90(description , _description , *description_len);
  free(description);
}
