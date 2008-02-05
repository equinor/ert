#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <util.h>
#include <hash.h>
#include <multz_config.h>
#include <enkf_config_node.h>
#include <path_fmt.h>
#include <ecl_static_kw_config.h>
#include <enkf_types.h>
#include <well_config.h>
#include <field_config.h>
#include <equil_config.h>
#include <multz_config.h>
#include <multflt_config.h>
#include <well_obs.h>
#include <pgbox_config.h>
#include <thread_pool.h>
#include <obs_node.h>
#include <obs_data.h>
#include <history.h>
#include <meas_matrix.h>
#include <enkf_state.h>  
#include <analysis.h>
#include <enkf_obs.h>
#include <sched_file.h>
#include <enkf_fs.h>
#include <void_arg.h>
#include <gen_kw_config.h>
#include <enkf_config.h>




struct enkf_config_struct {
  int  		   ens_size;
  int              iens_offset;
  hash_type       *config_hash;
  char            **well_list;
  int               Nwells;
  path_fmt_type    *run_path;
  path_fmt_type    *eclbase;
  bool              endian_swap;
  bool              fmt_file;
  bool              unified;
  char *            data_file;
};




enkf_impl_type enkf_config_impl_type(const enkf_config_type *enkf_config, const char * ecl_kw_name) {
  enkf_impl_type impl_type;

  if (hash_has_key(enkf_config->config_hash , ecl_kw_name)) {
    enkf_config_node_type * node = hash_get(enkf_config->config_hash , ecl_kw_name);
    impl_type = enkf_config_node_get_impl_type(node);
  } else
    impl_type = STATIC;

  return impl_type;
}



static void enkf_config_realloc_well_list(enkf_config_type * enkf_config) {
  enkf_config->well_list = realloc(enkf_config->well_list , enkf_config->Nwells * sizeof * enkf_config->well_list);
}


bool enkf_config_get_endian_swap(const enkf_config_type * enkf_config) { return enkf_config->endian_swap; }

bool enkf_config_get_fmt_file(const enkf_config_type * enkf_config) { return enkf_config->fmt_file; }

const char * enkf_config_get_data_file(const enkf_config_type * ens) { return ens->data_file; }


bool enkf_config_has_key(const enkf_config_type * config , const char * key) {
  return hash_has_key(config->config_hash , key);
}



enkf_config_type * enkf_config_alloc(int ens_size            , 
				     const char * data_file  , 
				     const char * _run_path  , 
				     const char * _eclbase   , 
				     bool fmt_file 	     ,
				     bool unified  	     ,         
				     bool endian_swap) {

  enkf_config_type * config = malloc(sizeof * config);
  config->config_hash = hash_alloc(10);
  config->ens_size      = ens_size;
  config->endian_swap   = endian_swap;
  config->Nwells        = 0;
  config->well_list     = NULL;  
  enkf_config_realloc_well_list(config);
  config->unified       = unified;
  config->fmt_file      = fmt_file;
  config->data_file     = util_alloc_string_copy(data_file);
  config->run_path     = path_fmt_alloc_directory_fmt(_run_path , true);
  config->eclbase      = path_fmt_alloc_file_fmt(_eclbase);

  return config;
}


const char ** enkf_config_get_well_list_ref(const enkf_config_type * ens , int *Nwells) {
  *Nwells = ens->Nwells;
  return (const char **) ens->well_list;
}


void enkf_config_add_well(enkf_config_type * enkf_config , const char *well_name , int size, const char ** var_list) {
  enkf_config_add_type(enkf_config , well_name , ecl_summary , WELL, NULL , well_config_alloc(well_name , size , var_list));
  enkf_config->Nwells++;
  enkf_config_realloc_well_list(enkf_config);
  enkf_config->well_list[enkf_config->Nwells - 1] = util_alloc_string_copy(well_name);
}


void enkf_config_add_gen_kw(enkf_config_type * enkf_config , const char * config_file) {
  enkf_config_add_type(enkf_config , "GEN_KW" , parameter , GEN_KW , NULL , gen_kw_config_fscanf_alloc(config_file , NULL));
}




void enkf_config_add_type(enkf_config_type * enkf_config , 
		       const char    * key      , 
		       enkf_var_type enkf_type  , 
		       enkf_impl_type impl_type , 
		       const char   * ecl_file  , 
		       const void   * data) {
  if (enkf_config_has_key(enkf_config , key)) {
    fprintf(stderr,"%s: a ensuration object:%s has already been added - aborting \n",__func__ , key);
    abort();
  }

  {
    config_free_ftype * freef;
    switch(impl_type) {
    case(FIELD):
      freef             = field_config_free__;
      break;
    case(MULTZ):
      freef             = multz_config_free__;
      break;
    case(WELL):
      freef             = well_config_free__;
      break;
    case(MULTFLT):
      freef             = multflt_config_free__;
      break;
    case(EQUIL):
      freef             = equil_config_free__;
      break;
    case(STATIC):
      freef             = ecl_static_kw_config_free__;
      break;
    case(PGBOX):
      freef             = pgbox_config_free__;
      break;
    case(GEN_KW):
      freef             = gen_kw_config_free__;
      break;
    default:
      fprintf(stderr,"%s : invalid implementation type: %d - aborting \n",__func__ , impl_type);
      abort();
    }
    {
      enkf_config_node_type * node = enkf_config_node_alloc(enkf_type , impl_type , key , ecl_file , data , freef);
      hash_insert_hash_owned_ref(enkf_config->config_hash , key , node , enkf_config_node_free__);
    }
  }
}



void enkf_config_add_type0(enkf_config_type * enkf_config , const char *key , int size, enkf_var_type enkf_type , enkf_impl_type impl_type) {
  switch(impl_type) {
  case(STATIC):
    enkf_config_add_type(enkf_config , key , enkf_type , impl_type , NULL , ecl_static_kw_config_alloc(size , key , key));
    break;
  case(FIELD):
    fprintf(stderr,"%s: Can not add FIELD ens objects like:%s on the run - these must be from the main program with enkf_config_add_type - sorry.\n",__func__ , key);
    abort();
    break;
  default:
    fprintf(stderr,"%s only STATIC and FIELD types are implemented - aborting \n",__func__);
    abort();
  }
}




const enkf_config_node_type * enkf_config_get_config_ref(const enkf_config_type * ens, const char * key) {
  if (hash_has_key(ens->config_hash , key)) {
    enkf_config_node_type * node = hash_get(ens->config_hash , key);
    return node;
  } else {
    fprintf(stderr,"%s: ens node:%s does not exist \n",__func__ , key);
    abort();
  }
}
