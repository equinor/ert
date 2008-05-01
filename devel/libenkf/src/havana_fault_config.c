#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <ens_config.h>
#include <gen_kw_config.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <assert.h>

#include <havana_fault_config.h>



/* Transform */
void havana_fault_config_transform(const havana_fault_config_type * config , const double * input_data , double * output_data) 
{
    gen_kw_config_transform(config->gen_kw_config ,  input_data ,  output_data);
}




havana_fault_config_type * havana_fault_config_fscanf_alloc(const char * filename , const char * template_file, const char * executable_file) 
{
    /* Allocate and read */
  if (!util_file_exists(filename) ) 
  {
      util_abort("%s: config_file:%s does not exist - aborting.\n" , __func__ , filename);
  }
  if (!util_file_exists(template_file) ) 
  {
      util_abort("%s: template_file:%s does not exist - aborting.\n" , __func__ , template_file);
  }
  if (!util_file_exists(executable_file)  ) 
  {
      util_abort("%s: executable_file:%s does not exist - aborting.\n" , __func__ , executable_file);
  }

  havana_fault_config_type *havana_fault_config = malloc(sizeof *havana_fault_config);

  havana_fault_config->havana_executable = util_alloc_string_copy(executable_file);
  havana_fault_config->gen_kw_config = gen_kw_config_fscanf_alloc(filename , template_file);
  return havana_fault_config;
}


void havana_fault_config_free(havana_fault_config_type * havana_fault_config) 
{
    assert(havana_fault_config);
    assert(havana_fault_config->gen_kw_config);
    gen_kw_config_type *tmp = havana_fault_config->gen_kw_config;
    gen_kw_config_free(tmp);
    havana_fault_config->gen_kw_config= NULL;

    if (havana_fault_config->havana_executable != NULL)
    {
        free(havana_fault_config->havana_executable);
    }
}

int havana_fault_config_get_data_size(const havana_fault_config_type * havana_fault_config) 
{
    return (gen_kw_config_get_data_size(havana_fault_config->gen_kw_config));
}


const char * havana_fault_config_get_name(const havana_fault_config_type * config, int kw_nr) 
{
    return (gen_kw_config_get_name(config->gen_kw_config,  kw_nr));
}


char ** havana_fault_config_get_name_list(const havana_fault_config_type * config) 
{
    return ( gen_kw_config_get_name_list(config->gen_kw_config));
}

const char * havana_fault_config_get_template_ref(const havana_fault_config_type * config) 
{
    return ( gen_kw_config_get_template_ref(config->gen_kw_config));
}

void  havana_fault_config_truncate(const havana_fault_config_type * config, scalar_type * scalar)
{
    util_abort("%s: the function is not implemented\n",__func__);
    /*    gen_kw_config_truncate(config->gen_kw_config, scalar); */
}




/*****************************************************************/

VOID_FREE(havana_fault_config)






