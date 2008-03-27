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
#include <enkf_types.h>
#include <analysis.h>
#include <enkf_obs.h>
#include <sched_file.h>
#include <enkf_fs.h>
#include <void_arg.h>
#include <gen_kw_config.h>
#include <enkf_config.h>
#include <ecl_grid.h>
#include <time.h>
#include <enkf_site_config.h>
#include <ecl_queue.h>
#include <lsf_driver.h>
#include <local_driver.h>


struct enkf_config_struct {
  ecl_grid_type    *grid;
  int  		    ens_size;
  int               ens_offset;
  int              iens_offset;
  hash_type       *config_hash;
  time_t            start_time;
  char            **well_list;
  int               Nwells;
  path_fmt_type    *run_path;
  path_fmt_type    *eclbase;
  path_fmt_type    *ecl_store_path;
  ecl_store_enum   *ecl_store;
  bool              endian_swap;
  bool              fmt_file;
  bool              unified;
  char *            data_file;
  char *            schedule_file;
  char *            obs_config_file;
  char *            ens_path;
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

bool enkf_config_get_unified(const enkf_config_type * enkf_config) { return enkf_config->unified; }

const char * enkf_config_get_data_file(const enkf_config_type * ens) { return ens->data_file; }

char ** enkf_config_alloc_keylist(const enkf_config_type * config , int *keys) {
  *keys = hash_get_size(config->config_hash);
  return hash_alloc_keylist(config->config_hash);
}



bool enkf_config_has_key(const enkf_config_type * config , const char * key) {
  return hash_has_key(config->config_hash , key);
}


void enkf_config_get_grid_dims(const enkf_config_type * config , int *nx , int *ny, int *nz , int *active_size) {
  ecl_grid_get_dims(config->grid , nx , ny , nz , active_size);
}


static void enkf_config_set_obs_config_file(enkf_config_type * config , const char * obs_config_file) {
  config->obs_config_file = util_realloc_string_copy(config->obs_config_file , obs_config_file);
}

static void enkf_config_set_ens_path(enkf_config_type * config , const char * ens_path) {
  config->ens_path = util_realloc_string_copy(config->obs_config_file , ens_path);
}

const char * enkf_config_get_ens_path(const enkf_config_type * config) {
  return config->ens_path;
}



static void enkf_config_set_eclbase(enkf_config_type * config , const char * eclbase) {
  if (config->eclbase != NULL)
    path_fmt_free(config->eclbase);
  config->eclbase = path_fmt_alloc_file_fmt(eclbase);
}


static void enkf_config_set_run_path(enkf_config_type * config , const char * run_path) {
  if (config->run_path != NULL)
    path_fmt_free(config->run_path);
  config->run_path = path_fmt_alloc_directory_fmt(run_path , true);
}


static void enkf_config_set_ecl_store_path(enkf_config_type * config , const char * ecl_store_path) {
  if (config->ecl_store_path != NULL)
    path_fmt_free(config->ecl_store_path);
  config->ecl_store_path = path_fmt_alloc_directory_fmt(ecl_store_path , false);
}

static void enkf_config_set_ecl_store(enkf_config_type * config , int store_value , int tokens, const char ** token_list) {
  if (config->ens_size <= 0) {
    fprintf(stderr,"%s: must set ens_size first - aborting \n",__func__);
    abort();
  }
  {
    int token_index   = 0;
    int prev_iens     = -1;
    bool range_active = false;
    do {
      if (token_list[token_index][0] == ',')
	token_index++;
      else {
	if (token_list[token_index][0] == '-') {
	  if (prev_iens == -1) {
	    fprintf(stderr,"%s: something rotten - lonesome dash \n",__func__);
	    abort();
	  }
	  range_active = true;
	} else {
	  int iens,iens1,iens2;
	  if (util_sscanf_int(token_list[token_index] , &iens2)) {
	    if (range_active)
	      iens1 = prev_iens;
	    else {
	      iens1     = iens2;
	      prev_iens = iens2;
	    }
	    if (iens1 >= config->ens_offset && iens2 < (config->ens_size + config->ens_offset)) {
	      for (iens = iens1; iens <= iens2; iens++) 
		config->ecl_store[iens - config->ens_offset] = store_value;
	    }
	    range_active = false;  
	  } else {
	    fprintf(stderr,"%s: something wrong when parsing: \"%s\" to integer \n",__func__ , token_list[token_index]);
	    abort();
	  }
	}
	token_index++;
      }
    } while (token_index < tokens);
  }
}



static void enkf_config_set_data_file(enkf_config_type * config , const char * data_file) {
  if (util_file_exists(data_file))
    config->data_file = util_realloc_string_copy(config->data_file , data_file);
  else {
    fprintf(stderr,"%s: sorry: data_file:%s does not exist - aborting \n",__func__ , data_file);
    abort();
  }
}

static void enkf_config_set_schedule_file(enkf_config_type * config , const char * schedule_file) {
  if (util_file_exists(schedule_file)) 
    config->schedule_file = util_realloc_string_copy(config->schedule_file , schedule_file);
  else {
    fprintf(stderr,"%s: sorry: schedule_file:%s does not exist - aborting \n",__func__ , schedule_file);
    abort();
  }
}

const char * enkf_config_get_schedule_file(const enkf_config_type * config) {
  return config->schedule_file;
}


const char * enkf_config_get_obs_config_file(const enkf_config_type * config) {
  return config->obs_config_file;
}


static void enkf_config_set_grid(enkf_config_type * config, const char * grid_file) {
  if (config->grid == NULL) 
    config->grid = ecl_grid_alloc(grid_file , config->endian_swap);
}

static void enkf_config_set_ens_size(enkf_config_type * config , int ens_size) {
  if (ens_size > 0) {
    int iens;
    config->ens_size  = ens_size;
    config->ecl_store = util_malloc(ens_size * sizeof * config->ecl_store , __func__);
    for (iens = 0; iens < ens_size; iens++)
      config->ecl_store[iens] = store_none;
  } else {
    fprintf(stderr,"%s: SIZE must be greater than zero - aborting \n",__func__);
    abort();
  }
}


/*
   1 MAY 1999
*/

static void enkf_config_set_start_date(enkf_config_type * config , const char ** tokens) {
  int day , year;
  bool OK = true;

  OK = util_sscanf_int(tokens[0] , &day);
  OK = OK && util_sscanf_int(tokens[2] , &year); 
  if (OK) {
    int month_nr = util_get_month_nr(tokens[1]);
    config->start_time = util_make_time1( day , month_nr , year );
  } else {
    fprintf(stderr,"%s: fatal error when parsing START_TIME \n",__func__);
    abort();
  }
}




static enkf_config_type * enkf_config_alloc_empty(int  ens_offset,
						  bool fmt_file ,
						  bool unified  ,         
						  bool endian_swap) {
  enkf_config_type * config = malloc(sizeof * config);
  config->config_hash = hash_alloc(10);

  config->endian_swap   = endian_swap;
  config->unified       = unified;
  config->fmt_file      = fmt_file;
  config->Nwells        = 0;
  config->well_list     = NULL;  
  enkf_config_realloc_well_list(config);

  /*
    All of these must be set before the config object is ready for use.
  */
  config->eclbase   	  = NULL;
  config->data_file 	  = NULL;
  config->grid      	  = NULL;
  config->run_path  	  = NULL;
  config->schedule_file   = NULL;
  config->start_time      = -1;
  config->ecl_store_path  = NULL;
  config->ens_size        = -1;
  config->ens_offset      = ens_offset;
  config->ecl_store       = NULL;
  config->obs_config_file = NULL;
  config->ens_path        = NULL;
  
  return config;
}



static void enkf_config_post_assert(const enkf_config_type * config) {

}

#define ASSERT_TOKENS(kw,t,n) if ((t - 1) < (n)) { fprintf(stderr,"%s: when parsing %s must have at least %d arguments - aborting \n",__func__ , kw , (n)); abort(); }
enkf_config_type * enkf_config_fscanf_alloc(const char * config_file , 
					    enkf_site_config_type * site_config , 
					    int  ens_offset,
					    bool fmt_file ,
					    bool unified  ,         
					    bool endian_swap) { 

  char * config_path;
  enkf_config_type * enkf_config = enkf_config_alloc_empty(ens_offset , fmt_file , unified , endian_swap);
  util_alloc_file_components(config_file , &config_path , NULL , NULL);
  FILE * stream = util_fopen(config_file , "r");
  char * line;
  bool at_eof = false;
  int * index_map = NULL;
  
  do {
    enkf_impl_type impl_type;
    int i , tokens;
    int active_tokens;
    char **token_list;

    line  = util_fscanf_alloc_line(stream , &at_eof);
    if (line != NULL) {
      util_split_string(line , " " , &tokens , &token_list);
      
      active_tokens = tokens;
      for (i = 0; i < tokens; i++) {
	if (token_list[i][0] == '-') {
	  if (token_list[i][1] == '-') {
	    active_tokens = i;
	    break;
	  }
	}
      }
      
      
      if (active_tokens > 0) {
	impl_type = enkf_types_check_impl_type(token_list[0]);
	if (impl_type == INVALID) {
	  
	  const char * kw = token_list[0];
	  if (enkf_site_config_has_key(site_config , kw)) {
	    /* The configuration overrides a value from the site_config object. */
	    char * site_value;
	    ASSERT_TOKENS(kw , active_tokens , 1);
	    site_value  = util_alloc_joined_string((const char **) &token_list[1] , active_tokens - 1 , " ");
	    enkf_site_config_set_key(site_config , kw , site_value);
	    free(site_value);
	  } else {
	    if (strcmp(kw , "SIZE") == 0) {
	      int ens_size;
	      ASSERT_TOKENS("SIZE" , active_tokens , 1);
	      if (util_sscanf_int(token_list[1] , &ens_size)) 
		enkf_config_set_ens_size( enkf_config , ens_size);
	      else {
		fprintf(stderr,"%s: failed to convert:%s to valid integer - aborting \n",__func__ , token_list[1]);
		abort();
	      }
	    } else if (strcmp(kw , "OBS_CONFIG") == 0) {
	      ASSERT_TOKENS("OBS_CONFIG" , active_tokens , 1);
	      {
		char * obs_config_file = util_alloc_full_path(config_path , token_list[1]);
		enkf_config_set_obs_config_file(enkf_config , obs_config_file);
		free(obs_config_file);
	      }
	    } else if (strcmp(kw , "ENSPATH") == 0) {
	      ASSERT_TOKENS("ENSPATH" , active_tokens , 1);
	      enkf_config_set_ens_path(enkf_config , token_list[1]);
	    } else if (strcmp(kw , "RUNPATH") == 0) {
	      ASSERT_TOKENS("RUNPATH" , active_tokens , 1);
	      enkf_config_set_run_path( enkf_config , token_list[1] );
	    } else if (strcmp(kw , "DATA_FILE") == 0) {
	      ASSERT_TOKENS("DATA_FILE" , active_tokens , 1);
	      enkf_config_set_data_file( enkf_config , token_list[1] );
	    } else if (strcmp(kw , "ECLBASE") == 0) {
	      ASSERT_TOKENS("ECLBASE" , active_tokens , 1);
	      enkf_config_set_eclbase( enkf_config , token_list[1] );
	    } else if (strcmp(kw , "SCHEDULE_FILE") == 0) {
	      ASSERT_TOKENS("SCHEDULE_FILE" , active_tokens , 1);
	      if (enkf_config->start_time == -1) {
		fprintf(stderr,"%s: must set START_TIME before SCHEDULE_FILE - aborting \n",__func__);
		abort();
	      }
	      enkf_config_set_schedule_file(enkf_config , token_list[1]);
	    } else if (strcmp(kw , "ECL_STORE_PATH") == 0) {
	      ASSERT_TOKENS("ECL_STORE_PATH" , active_tokens , 1);
	      enkf_config_set_ecl_store_path(enkf_config , token_list[1]);
	    } else if (strcmp(kw , "ECL_STORE") == 0) {
	      ASSERT_TOKENS("ECL_STORE" , active_tokens , 2);
	      int ecl_store;
	      if (enkf_config->ecl_store_path == NULL) {
		fprintf(stderr,"%s: must configure ECL_STORE_PATH prior to ECL_STORE - aborting \n",__func__);
		abort();
	      } else {
		if (util_sscanf_int(token_list[1] , &ecl_store)) 
		  enkf_config_set_ecl_store(enkf_config , ecl_store , active_tokens - 2 , (const char **) &token_list[2]);
		else {
		  fprintf(stderr,"%s: error when parsing: %s to integer - aborting \n",__func__ , token_list[1]);
		  abort();
		}
	      }
	    } else if (strcmp(kw , "GRID") == 0) {
	      ASSERT_TOKENS("GRID" , active_tokens , 1);
	      enkf_config_set_grid(enkf_config , token_list[1]);
	    } else if (strcmp(kw , "START_TIME") == 0) {
	      ASSERT_TOKENS("START_TIME" , active_tokens , 3);
	      enkf_config_set_start_date(enkf_config , (const char **) &token_list[1]);
	    } else
	      fprintf(stderr,"%s: ** Warning ** keyword: %s not recognzied - line ignored \n",__func__ , kw);
	  }    
	} else {
	  switch(impl_type) {
	  case(MULTZ):
	    {
	      const char * key         = token_list[1];
	      const char * ecl_file    = token_list[2];
	      char       * config_file = util_alloc_full_path(config_path , token_list[3]);
	      int   nx,ny,nz,active_size;
		
	      if (enkf_config->grid == NULL) {
		fprintf(stderr,"%s must add grid prior to adding MULTZ - aborting \n",__func__);
		abort();
	      }
	      ecl_grid_get_dims(enkf_config->grid , &nx , &ny , &nz , &active_size);
	      enkf_config_add_type(enkf_config , key , parameter , MULTZ , ecl_file , multz_config_fscanf_alloc(config_file , nx , ny , nz));
	      free(config_file);
	    }
	    break;
	  case(MULTFLT):
	    {
	      const char * key         = token_list[1];
	      const char * ecl_file    = token_list[2];
	      char       * config_file = util_alloc_full_path(config_path , token_list[3]);
	      enkf_config_add_type(enkf_config , key , parameter , MULTFLT , ecl_file , multflt_config_fscanf_alloc(config_file));
	      free(config_file);
	    }
	    break;
	  case(EQUIL):
	    {
	      const char * key         = token_list[1];
	      const char * ecl_file    = token_list[2];
	      char       * config_file = util_alloc_full_path(config_path , token_list[3]);
	      enkf_config_add_type(enkf_config , key , parameter , EQUIL , ecl_file , equil_config_fscanf_alloc(config_file));
	      free(config_file);
	    }
	    break;
	  case(FIELD):
	    ASSERT_TOKENS("FIELD" , active_tokens , 2);
	    {
	      const char * key             = token_list[1];
	      const char * var_type_string = token_list[2];
	      int   nx,ny,nz,active_size;

	      if (enkf_config->grid == NULL) {
		fprintf(stderr,"%s must add grid prior to adding FIELD - aborting \n",__func__);
		abort();
	      }
	      ecl_grid_get_dims(enkf_config->grid , &nx , &ny , &nz , &active_size);
	      if (index_map == NULL) index_map = (int *) ecl_grid_alloc_index_map(enkf_config->grid);
		
	      if (strcmp(var_type_string , "DYNAMIC") == 0)
		enkf_config_add_type(enkf_config , key , ecl_restart , FIELD , NULL , field_config_alloc_dynamic(key , nx , ny , nz , active_size , index_map));
	      else if (strcmp(var_type_string , "PARAMETER") == 0) {
		ASSERT_TOKENS("FIELD" , active_tokens , 5);
		{
		  const char * ecl_file = token_list[3];
		  int init_mode;
		  if (util_sscanf_int(token_list[4] , &init_mode)) 
		    enkf_config_add_type(enkf_config , key , parameter   , FIELD , ecl_file , field_config_alloc_parameter(key , 
															   nx , ny , nz , active_size , 
															   index_map , 
															   0 , init_mode , active_tokens - 5 , (const char **) &token_list[5]));
		  else {
		    fprintf(stderr,"%s: init mode must be valid int - aborting \n",__func__);
		    abort();
		  }
		}
	      } else {
		fprintf(stderr,"%s : aborting \n",__func__);
		abort();
	      }
	    }
	    break;
	  case(WELL):
	    ASSERT_TOKENS("WELL" , active_tokens , 2);
	    enkf_config_add_well(enkf_config , token_list[1] , active_tokens - 2 , (const char **) &token_list[2]);
	    break;
	  case(PGBOX):
	    break;
	  case(GEN_KW):
	    break;
	  default:
	    fprintf(stderr,"%s: Invalid keyword: %s - aborting \n",__func__ ,  token_list[0]);
	    abort();
	  }
	}
      }
      util_free_string_list(token_list , tokens);
      free(line);
    }
  } while (!at_eof);
  if (config_path != NULL) free(config_path);
  enkf_config_post_assert(enkf_config);
  enkf_site_config_validate(site_config);
  return enkf_config;
}
#undef ASSERT_TOKENS



/*
enkf_config_type * enkf_config_alloc(int ens_size            , 
				     const int  * start_date ,
				     const char * grid_file  , 
				     const char * data_file  , 
				     const char * _run_path  , 
				     const char * _eclbase) {

  enkf_config_type * config = malloc(sizeof * config);
  config->ens_size      = ens_size;
  config->grid          = ecl_grid_alloc(grid_file , endian_swap);
  config->data_file    = util_alloc_string_copy(data_file);
  config->run_path     = path_fmt_alloc_directory_fmt(_run_path , true);
  config->eclbase      = path_fmt_alloc_file_fmt(_eclbase);
  memcpy(config->start_date , start_date , 3 * sizeof * start_date);
  return config;
}
*/


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
    fprintf(stderr,"%s: a configuration object:%s has already been added - aborting \n",__func__ , key);
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



void enkf_config_add_field_config(enkf_config_type * enkf_config) {
  
}


const enkf_config_node_type * enkf_config_get_node_ref(const enkf_config_type * ens, const char * key) {
  if (hash_has_key(ens->config_hash , key)) {
    enkf_config_node_type * node = hash_get(ens->config_hash , key);
    return node;
  } else {
    fprintf(stderr,"%s: ens node:%s does not exist \n",__func__ , key);
    abort();
  }
}




char * enkf_config_alloc_ecl_store_path(const enkf_config_type * config , int iens) {
  if (config->ecl_store_path != NULL)
    return path_fmt_alloc_path(config->ecl_store_path , iens);
  else
    return NULL;
}

char * enkf_config_alloc_run_path(const enkf_config_type * config , int iens) {
  return path_fmt_alloc_path(config->run_path , iens);
}

char * enkf_config_alloc_eclbase(const enkf_config_type * config , int iens) {
  return path_fmt_alloc_path(config->eclbase , iens);
}

int    	       enkf_config_get_ens_size  (const enkf_config_type * config) { return config->ens_size; }
int    	       enkf_config_get_ens_offset(const enkf_config_type * config) { return config->ens_offset; }
time_t 	       enkf_config_get_start_date(const enkf_config_type * config) { return config->start_time; }
ecl_store_enum enkf_config_iget_ecl_store(const enkf_config_type * config, int iens) { return config->ecl_store[iens]; }



void enkf_config_free(enkf_config_type * config) {  
  hash_free(config->config_hash);
  {
    int i;
    for (i=0; i < config->Nwells; i++)
      free(config->well_list[i]);
    free(config->well_list);
  }
  path_fmt_free(config->run_path);
  path_fmt_free(config->eclbase);
  ecl_grid_free(config->grid);
  free(config->data_file);
  if (config->ecl_store_path != NULL)
    path_fmt_free(config->ecl_store_path);
  if (config->obs_config_file != NULL) free(config->obs_config_file);
  if (config->ens_path != NULL) free(config->ens_path);
  free(config->ecl_store);
  free(config);
}



ecl_queue_type * enkf_config_alloc_ecl_queue(const enkf_config_type * config , const enkf_site_config_type * site_config) {
  ecl_queue_type          * ecl_queue;
  int                       max_running;  
  basic_queue_driver_type * queue_driver;

  const char * queue_system = enkf_site_config_get_value(site_config , "QUEUE_SYSTEM");
  if (strcmp(queue_system , "LSF") == 0) {
    const char * resource_request = enkf_site_config_get_value(site_config , "LSF_RESOURCES");
    const char * queue_name       = enkf_site_config_get_value(site_config , "LSF_QUEUE");
    max_running  = strtol(enkf_site_config_get_value(site_config , "MAX_RUNNING_LSF") , NULL , 10);
    queue_driver = lsf_driver_alloc(queue_name , resource_request);
  } else if (strcmp(queue_system , "LOCAL") == 0) {
    queue_driver = local_driver_alloc();
    max_running  = strtol(enkf_site_config_get_value(site_config , "MAX_RUNNING_LOCAL") , NULL , 10);
  }
  else {
    fprintf(stderr,"%s: internal error - queue_system:%s not recognized - aborting \n",__func__ , queue_system);
    abort();
  }
  
  {
    int max_submit  = 2;
    const char * eclipse_LD_path;
    const char * __run_path = path_fmt_get_fmt(config->run_path);
    const char * __ecl_base = path_fmt_get_fmt(config->eclbase);
    char       * __target_file_fmt;
    char  restart_extension[7];
    path_fmt_type * target_file_fmt;
    
    if (enkf_config_get_fmt_file(config))
      sprintf(restart_extension , ".F%s04d" , "%");
    else
      sprintf(restart_extension , ".X%s04d" , "%");
    __target_file_fmt = util_alloc_joined_string( (const char *[4]) {__run_path , UTIL_PATH_SEP , __ecl_base , restart_extension} , 4 , "");
    target_file_fmt = path_fmt_alloc_file_fmt(__target_file_fmt);

    if (enkf_site_config_node_set(site_config , "ECLIPSE_LD_PATH"))
      eclipse_LD_path = enkf_site_config_get_value(site_config , "ECLIPSE_LD_PATH");
    else
      eclipse_LD_path = NULL;
    
    ecl_queue = ecl_queue_alloc(enkf_config_get_ens_size(config),
				max_running , 
				max_submit  ,
				enkf_site_config_get_value(site_config , "START_ECLIPSE_CMD"),
				enkf_site_config_get_value(site_config , "ECLIPSE_EXECUTABLE"),
				eclipse_LD_path , 
				enkf_site_config_get_value(site_config , "ECLIPSE_CONFIG"),
				enkf_site_config_get_value(site_config , "LICENSE_SERVER"),
				config->run_path , 
				config->eclbase , 
				target_file_fmt  , 
				queue_driver);
    
    path_fmt_free(target_file_fmt);
    free(__target_file_fmt);
  }
  return ecl_queue;
}
