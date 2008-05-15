#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <ens_config.h>
#include <gen_kw_config.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <trans_func.h>
#include <scalar_config.h>




static gen_kw_config_type * __gen_kw_config_alloc_empty(int size, const char * template_file) {
  gen_kw_config_type *gen_kw_config = malloc(sizeof *gen_kw_config);
  gen_kw_config->kw_list       = enkf_util_malloc(size * sizeof *gen_kw_config->kw_list , __func__);
  gen_kw_config->scalar_config = scalar_config_alloc_empty(size);
  gen_kw_config->var_type      = parameter;
  gen_kw_config->template_file = util_alloc_string_copy(template_file);
  gen_kw_config->executable    = NULL;

  /* 
     Allows for template_file == NULL - as a way to adapt to
     the requirements of the havana_fault object. Should maybe
     not bend-over in this way? 
  */
   
  if (template_file != NULL) {
    if (!util_file_exists(template_file))
      util_abort("%s: the template_file:%s does not exist - aborting.\n",__func__ , template_file);
  }
  return gen_kw_config;
}




void gen_kw_config_transform(const gen_kw_config_type * config , const double * input_data , double * output_data) {
  scalar_config_transform(config->scalar_config , input_data , output_data);
}



/**
This function will allocate a gen_kw_config keyword. The first
argument is the name of a file containing the keywords, and the second
argument is the name of the template file used.

The format of the file containing keywords is as follows:
  ________________________
 /
 | KEY1  UNIFORM 0     1
 | KEY2  NORMAL  10   10
 | KEY2  CONST   0.25
 \________________________

The first part is just the keyword, the second part is the properties
of the prior distribution of that keyword. That is implemented as an
object of type scalar/scalar_config - and documented there.

For the template file there are essentially no restrictions:

 o All occurences of <KEY1> are replaced with the corresponding value.

 o The file template file must exist when the function
   gen_kw_config_fscanf_alloc() is called.

*/

gen_kw_config_type * gen_kw_config_fscanf_alloc(const char * filename , const char * template_file) {
  gen_kw_config_type * config;
  if (util_file_exists(filename)) {
    FILE * stream = util_fopen(filename , "r");
    int line_nr = 0;
    int size;
    
    size = util_count_file_lines(stream);
    fseek(stream , 0L , SEEK_SET);
    config = __gen_kw_config_alloc_empty(size , template_file);
    do {
      char name[128];  /* UGGLY HARD CODED LIMIT */
      if (fscanf(stream , "%s" , name) != 1) {
	fprintf(stderr,"%s: something wrong when reading: %s - aborting \n",__func__ , filename);
	abort();
      }
      config->kw_list[line_nr] = util_alloc_string_copy(name);
      scalar_config_fscanf_line(config->scalar_config , line_nr , stream);
      line_nr++;
    } while ( line_nr < size );
    fclose(stream);
  }
  else 
    util_abort("%s: config_file:%s does not exist - aborting.\n" , __func__ , filename);

  return config;
}


void gen_kw_config_free(gen_kw_config_type * gen_kw_config) {
  util_free_string_list(gen_kw_config->kw_list , scalar_config_get_data_size(gen_kw_config->scalar_config));
  if (gen_kw_config->template_file != NULL)
    free(gen_kw_config->template_file);
  scalar_config_free(gen_kw_config->scalar_config);
  free(gen_kw_config);
}


int gen_kw_config_get_data_size(const gen_kw_config_type * gen_kw_config) {
  return scalar_config_get_data_size(gen_kw_config->scalar_config);
}


const char * gen_kw_config_get_name(const gen_kw_config_type * config, int kw_nr) {
  const int size = gen_kw_config_get_data_size(config);
  if (kw_nr >= 0 && kw_nr < size) 
    return config->kw_list[kw_nr];
  else {
    fprintf(stderr,"%s: asked for kw number:%d - valid interval: [0,%d] - aborting \n",__func__ , kw_nr , size - 1);
    abort();
  }
}


char ** gen_kw_config_get_name_list(const gen_kw_config_type * config) {
  return config->kw_list;
}


const char * gen_kw_config_get_template_ref(const gen_kw_config_type * config) {
  return config->template_file;
}


/*****************************************************************/

VOID_FREE(gen_kw_config)
