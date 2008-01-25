#include <gen_kw.h>
#include <gen_kw_config.h>
#include <util.h>

static gen_kw_config_type  * GEN_KW_CONFIG = NULL;
static gen_kw_type        ** GEN_KW_LIST   = NULL;

/*****************************************************************/

void gen_kw_inter_init__(const char * _config_file, const int * config_file_len , const char * _template_file , const int * template_file_len , const int * ens_size, const int * n_gen_kw) {
  int iens;
  char * config_file   = util_alloc_cstring(_config_file , config_file_len);
  char * template_file = util_alloc_cstring(_template_file , template_file_len);
  GEN_KW_CONFIG = gen_kw_config_fscanf_alloc(config_file , template_file);
  
  if (*n_gen_kw != gen_kw_config_get_data_size(GEN_KW_CONFIG)) {
    fprintf(stderr,"%s: size mismatch config_file:%d  mod_dimensions.F90/num_gen_kw:%d - aborting \n",__func__ , gen_kw_config_get_data_size(GEN_KW_CONFIG) , *n_gen_kw);
    abort();
  }


  GEN_KW_LIST = malloc(*ens_size * sizeof * GEN_KW_LIST);
  
  for (iens = 0; iens < *ens_size; iens++) 
    GEN_KW_LIST[iens] = gen_kw_alloc(GEN_KW_CONFIG);
  
  free(template_file);
  free(config_file);
}






void gen_kw_inter_ecl_write_avg__(const int * ens_size , const char *_path , const int * path_len) {
  char *path = util_alloc_cstring(_path , path_len);
  
  gen_kw_type *avg_gen_kw = gen_kw_alloc_mean(*ens_size , (const gen_kw_type **) GEN_KW_LIST);
  
  /*
    if (*avg_output == 1)
    gen_kw_direct_ecl_write(avg_gen_kw , path);
    else
    gen_kw_ecl_write(avg_gen_kw , path);
  */

  gen_kw_free(avg_gen_kw);
  free(path);
}



void gen_kw_inter_transform_gen_kw_data__(const int * iens , const double * input_data , double * output_data) {
  gen_kw_set_data(GEN_KW_LIST[(*iens) - 1] , input_data);
  gen_kw_output_transform(GEN_KW_LIST[(*iens) - 1]);
  gen_kw_get_output_data(GEN_KW_LIST[(*iens) - 1] , output_data);
}



void gen_kw_get_data__(const int * iens, double * data) {
  gen_kw_get_data(GEN_KW_LIST[(*iens) - 1] , data);
}


void gen_kw_inter_sample__(const int * iens, double * data) {
  gen_kw_sample(GEN_KW_LIST[(*iens) - 1]);
  gen_kw_get_data(GEN_KW_LIST[(*iens) - 1] , data);
}


void gen_kw_inter_truncate__(const int * iens , double * data) {
  gen_kw_set_data(GEN_KW_LIST[(*iens) - 1] , data);
  gen_kw_truncate(GEN_KW_LIST[(*iens) - 1]);
  gen_kw_get_data(GEN_KW_LIST[(*iens) - 1] , data);
}



void gen_kw_inter_get_description__(const int * gen_kw_nr , char * _description , const int * description_len) {
  const char * description = gen_kw_get_name(GEN_KW_LIST[0] , (*gen_kw_nr) - 1);
  util_memcpy_string_C2f90(description , _description , *description_len);
}



void gen_kw_output_filter__(const char * _output_file , const int * output_file_len , const double * data , const int * iens) {
  char * output_file = util_alloc_cstring(_output_file , output_file_len);

  if (gen_kw_config_get_data_size(GEN_KW_CONFIG) > 0)
    gen_kw_set_data(GEN_KW_LIST[(*iens) - 1]  , data);
  
  gen_kw_filter_file(GEN_KW_LIST[(*iens) - 1] , output_file);

  free(output_file);
}
