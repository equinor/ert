#include <stdlib.h>
#include <string.h>
#include <util.h>
#include <hash.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <trans_func.h>
#include <scalar_config.h>
#include <enkf_defaults.h>
#include <gen_kw_common.h>
#include <gen_kw_config.h>
#include <path_fmt.h>

#define GEN_KW_CONFIG_TYPE_ID 550761

struct gen_kw_config_struct {
  UTIL_TYPE_ID_DECLARATION;
  char                * key;
  char               ** kw_list;
  char               ** tagged_kw_list;  /* The same keywords - but '<' and '>' */
  scalar_config_type  * scalar_config;
  char                * template_file;
  gen_kw_type         * min_std;
  path_fmt_type       * init_file_fmt;   /* The format for loading init_files - if this is NULL the initialization is done by sampling N(0,1) numbers. */
};




void gen_kw_config_set_init_file_fmt( gen_kw_config_type * gen_kw_config , const char * init_file_fmt ) {
  if (gen_kw_config->init_file_fmt != NULL)
    path_fmt_free( gen_kw_config->init_file_fmt );
  
  if (init_file_fmt != NULL)
    gen_kw_config->init_file_fmt = path_fmt_alloc_path_fmt( init_file_fmt );
  else
    gen_kw_config->init_file_fmt = NULL;
}


char * gen_kw_config_alloc_initfile( const gen_kw_config_type * gen_kw_config , int iens ) {
  if (gen_kw_config->init_file_fmt != NULL)
    return path_fmt_alloc_path( gen_kw_config->init_file_fmt , false , iens);
  else
    return NULL;
}


static gen_kw_config_type * __gen_kw_config_alloc_empty(int size, const char * template_file, const char * init_file_fmt) {
  gen_kw_config_type *gen_kw_config = util_malloc(sizeof *gen_kw_config , __func__);
  UTIL_TYPE_ID_INIT(gen_kw_config , GEN_KW_CONFIG_TYPE_ID);
  gen_kw_config->kw_list            = util_malloc(size * sizeof *gen_kw_config->kw_list , __func__);
  gen_kw_config->tagged_kw_list     = util_malloc(size * sizeof *gen_kw_config->tagged_kw_list , __func__);
  gen_kw_config->scalar_config      = scalar_config_alloc_empty(size);
  gen_kw_config->template_file      = util_alloc_string_copy(template_file);
  gen_kw_config->min_std            = NULL;
  gen_kw_config->key                = NULL; 
  gen_kw_config->init_file_fmt      = NULL;
  
  gen_kw_config_set_init_file_fmt( gen_kw_config , init_file_fmt );
  if (!util_file_exists(template_file))
    util_abort("%s: the template_file:%s does not exist - aborting.\n",__func__ , template_file);
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

OPTIONS:

MIN_STD:
INIT_FILES:

*/

gen_kw_config_type * gen_kw_config_fscanf_alloc(const char * key , const char * filename , const char * template_file, const stringlist_type * options) {
  char * min_std_file  = NULL;
  char * init_file_fmt = NULL;
  gen_kw_config_type * config = NULL;

  /* Starting on the options. */
  {
    hash_type * opt_hash = hash_alloc_from_options( options );
    hash_iter_type * iter = hash_iter_alloc(opt_hash);
    const char * option = hash_iter_get_next_key(iter);
    while (option != NULL) {
      const char * value = hash_get( opt_hash , option );
      /*
	This could (should ??) have been implemented with
	if-then-else; isolated if-blocks have been chosen for
	clarity. Must update option_OK in every block, and check it
	at the bottom.

      */

      if (strcmp(option , "MIN_STD") == 0) 
	min_std_file = util_alloc_string_copy( value );
      else if (strcmp( option , "INIT_FILES") == 0)
        init_file_fmt = util_alloc_string_copy( value );
      else
        fprintf(stderr,"** Warning: options:%s not recognized \n",option);
      
      option = hash_iter_get_next_key(iter);
    }
    hash_iter_free( iter );
    hash_free( opt_hash );
  }


  if (util_file_exists(filename)) {
    FILE * stream = util_fopen(filename , "r");
    int line_nr = 0;
    int size;
    
    size = util_count_file_lines(stream);
    fseek(stream , 0L , SEEK_SET);
    config = __gen_kw_config_alloc_empty(size , template_file , init_file_fmt);
    do {
      char name[128];  /* UGGLY HARD CODED LIMIT */
      if (fscanf(stream , "%s" , name) != 1) 
        util_abort("%s: something wrong when reading: %s - aborting \n",__func__ , filename);
      
      config->tagged_kw_list[line_nr] = util_alloc_sprintf("%s%s%s" , DEFAULT_START_TAG , name , DEFAULT_END_TAG);
      config->kw_list[line_nr] = util_alloc_string_copy(name);
      scalar_config_fscanf_line(config->scalar_config , line_nr , stream);
      line_nr++;
    } while ( line_nr < size );
    fclose(stream);
  } else 
    util_abort("%s: config_file:%s does not exist - aborting.\n" , __func__ , filename);
  
  config->key = util_alloc_string_copy( key );
  if (min_std_file != NULL) {
    config->min_std = gen_kw_alloc( config );
    gen_kw_fload( config->min_std , min_std_file );
  }
  
  return config;
}


gen_kw_type * gen_kw_config_get_min_std( const gen_kw_config_type * gen_kw_config ) {
  return gen_kw_config->min_std;
}


void gen_kw_config_free(gen_kw_config_type * gen_kw_config) {
  util_free_stringlist(gen_kw_config->kw_list        , scalar_config_get_data_size(gen_kw_config->scalar_config));
  util_free_stringlist(gen_kw_config->tagged_kw_list , scalar_config_get_data_size(gen_kw_config->scalar_config));
  util_safe_free( gen_kw_config->key );
  if (gen_kw_config->template_file != NULL)
    free(gen_kw_config->template_file);
  scalar_config_free(gen_kw_config->scalar_config);
  if (gen_kw_config->init_file_fmt != NULL)
    path_fmt_free( gen_kw_config->init_file_fmt );
  free(gen_kw_config);
}



int gen_kw_config_get_data_size(const gen_kw_config_type * gen_kw_config) {
  return scalar_config_get_data_size(gen_kw_config->scalar_config);
}



const char * gen_kw_config_get_key(const gen_kw_config_type * config ) {
  return config->key;
}


char * gen_kw_config_alloc_user_key(const gen_kw_config_type * config , const char * key , int kw_nr) {
  char * user_key = util_alloc_sprintf("%s:%s" , key ,gen_kw_config_iget_name( config , kw_nr ));
  return user_key;
}


const char * gen_kw_config_iget_name(const gen_kw_config_type * config, int kw_nr) {
  const int size = gen_kw_config_get_data_size(config);
  if (kw_nr >= 0 && kw_nr < size) 
    return config->kw_list[kw_nr];
  else {
    util_abort("%s: asked for kw number:%d - valid interval: [0,%d] - aborting \n",__func__ , kw_nr , size - 1);
    return NULL;
  }
}




const char * gen_kw_config_get_tagged_name(const gen_kw_config_type * config, int kw_nr) {
  const int size = gen_kw_config_get_data_size(config);
  if (kw_nr >= 0 && kw_nr < size) 
    return config->tagged_kw_list[kw_nr];
  else {
    util_abort("%s: asked for kw number:%d - valid interval: [0,%d] - aborting \n",__func__ , kw_nr , size - 1);
    return NULL;
  }
}


char ** gen_kw_config_get_name_list(const gen_kw_config_type * config) {
  return config->kw_list;
}


const char * gen_kw_config_get_template_ref(const gen_kw_config_type * config) {
  return config->template_file;
}


const scalar_config_type * gen_kw_config_get_scalar_config(const gen_kw_config_type * config) {
  return config->scalar_config;
}


/**
   Will return -1 if the index is invalid.
*/
int gen_kw_config_get_index(const gen_kw_config_type * config , const char * key) {
  const int size   = gen_kw_config_get_data_size(config);
  bool    have_key = false;
  int     index    = 0;
  
  while (index < size && !have_key) {
    if (strcmp(config->kw_list[index] , key) == 0)
      have_key = true;
    else
      index++;
  }
  
  if (have_key)
    return index;
  else
    return -1;
}




/*****************************************************************/

SAFE_CAST(gen_kw_config , GEN_KW_CONFIG_TYPE_ID)
VOID_FREE(gen_kw_config)
VOID_GET_DATA_SIZE(gen_kw)
