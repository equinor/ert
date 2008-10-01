#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>
#include <util.h>
#include <hash.h>
#include <set.h>
#include <multz_config.h>
#include <enkf_config_node.h>
#include <path_fmt.h>
#include <enkf_types.h>
#include <well_config.h>
#include <field_config.h>
#include <equil_config.h>
#include <gen_param_config.h>
#include <multflt_config.h>
#include <well_obs.h>
#include <thread_pool.h>
#include <obs_node.h>
#include <obs_data.h>
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
#include <job_queue.h>
#include <lsf_driver.h>
#include <local_driver.h>
#include <rsh_driver.h>
#include <relperm_config.h>
#include <summary.h>
#include <summary_config.h>
#include <havana_fault_config.h>
#include <ext_joblist.h>
#include <gen_data.h>
#include <gen_data_config.h>
#include <stringlist.h>



struct enkf_config_struct {
  ecl_grid_type   *grid;
  int  		   ens_size;
  hash_type       *config_hash;
  hash_type       *data_kw;   /* This is just temporary during the init process - is later transfered to the individual enkf_state objects */
  time_t           start_time;
  path_fmt_type    *result_path;
  path_fmt_type    *run_path;
  path_fmt_type    *eclbase;
  bool             *keep_runpath;
  bool              endian_swap;
  bool              fmt_file;
  bool              unified;
  int               debug_level;
  char *            data_file;
  char *            schedule_src_file;
  char *            schedule_target_file;
  char *            obs_config_file;
  char *            ens_path;
  char *            init_file;
  char *            enkf_sched_file;
  stringlist_type  *forward_model;
  bool              include_all_static_kw;  /* If true all static keywords are stored.*/ 
  set_type         *static_kw_set;          /* Minimum set of static keywords which must be included to make valid restart files. */
};




enkf_impl_type enkf_config_impl_type(const enkf_config_type *enkf_config, const char * ecl_kw_name) {
  enkf_impl_type impl_type = INVALID;

  if (hash_has_key(enkf_config->config_hash , ecl_kw_name)) {
    enkf_config_node_type * node = hash_get(enkf_config->config_hash , ecl_kw_name);
    impl_type = enkf_config_node_get_impl_type(node);
  } else
    util_abort("%s: internal error: asked for implementation type of unknown node:%s \n",__func__ , ecl_kw_name);

  return impl_type;
}


enkf_var_type enkf_config_var_type(const enkf_config_type *enkf_config, const char * ecl_kw_name) {
  enkf_var_type var_type = invalid;

  if (hash_has_key(enkf_config->config_hash , ecl_kw_name)) {
    enkf_config_node_type * node = hash_get(enkf_config->config_hash , ecl_kw_name);
    var_type = enkf_config_node_get_var_type(node);
  } else
    util_abort("%s: internal error: asked for implementation type of unknown node:%s \n",__func__ , ecl_kw_name);

  return var_type;
}


int  enkf_config_get_debug(const enkf_config_type * enkf_config)             { return enkf_config->debug_level; }
void enkf_config_set_debug(enkf_config_type * enkf_config , int debug_level) { enkf_config->debug_level = debug_level; }

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
  config->ens_path = util_realloc_string_copy(config->ens_path , ens_path);
}

const char * enkf_config_get_ens_path(const enkf_config_type * config) {
  return config->ens_path;
}


static void enkf_config_set_enkf_sched_file(enkf_config_type * config , const char * enkf_sched_file) {
  if (util_file_exists(enkf_sched_file))
    config->enkf_sched_file = util_realloc_string_copy( config->enkf_sched_file , enkf_sched_file);
  else
    util_abort("%s: can not find file: %s \n", __func__ , enkf_sched_file);
}


const char * enkf_config_get_enkf_sched_file(const enkf_config_type * config) {
  return config->enkf_sched_file;
}

static void enkf_config_set_init_file(enkf_config_type * config , const char * init_file) {
  if (config->init_file != NULL)
    free(config->init_file);

  if (! util_file_exists(init_file)) 
    util_abort("%s: Sorry EQUIL_INIT_FILE points to:%s which can not be found. \n",__func__ , init_file);

  config->init_file = util_alloc_realpath(init_file);
}

const char * enkf_config_get_init_file(const enkf_config_type * config) {
  return config->init_file;
}



/** 
    The functions enkf_config_get_data_kw() and
    enkf_config_alloc_data_kw_list() are small helper functions for
    the enkf_config's handling of data_kw from the main configuration
    file.  

    The point is that under operation these keywords are owned (and
    used) by the enkf_state objects and not by the enkf_config
    objects. Maybe a bit awkward, because there is (currently) no way
    to get state specific information in (but - that might come ....).
*/



void * enkf_config_get_data_kw(const enkf_config_type * config , const char * key) {
  return hash_get(config->data_kw , key);
}


char ** enkf_config_alloc_data_kw_key_list(const enkf_config_type * config, int * size) {
  *size = hash_get_size(config->data_kw);
  return hash_alloc_keylist(config->data_kw);
}


static void enkf_config_set_eclbase(enkf_config_type * config , const char * eclbase) {
  if (config->eclbase != NULL)
    path_fmt_free(config->eclbase);
  config->eclbase = path_fmt_alloc_path_fmt(eclbase);
}


void enkf_config_set_run_path(enkf_config_type * config , const char * run_path) {
  if (config->run_path != NULL)
    path_fmt_free(config->run_path);
  config->run_path = path_fmt_alloc_directory_fmt(run_path);
}


static void enkf_config_set_result_path(enkf_config_type * config , const char * result_path) {
  if (config->result_path != NULL)
    path_fmt_free(config->result_path);
  config->result_path = path_fmt_alloc_directory_fmt(result_path);
}


static void enkf_config_set_ecl_store(enkf_config_type * config , bool keep_runpath , int tokens, const char ** token_list) {
  if (config->ens_size <= 0) 
    util_abort("%s: must set ens_size first - aborting \n",__func__);
  
  {
    int token_index   = 0;
    int prev_iens     = -1;
    bool range_active = false;
    do {
      if (token_list[token_index][0] == ',')
	token_index++;
      else {
	if (token_list[token_index][0] == '-') {
	  if (prev_iens == -1) 
	    util_abort("%s: something rotten - lonesome dash \n",__func__);

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
	    for (iens = iens1; iens <= iens2; iens++)
	      if (iens2 < config->ens_size) 
		config->keep_runpath[iens] = keep_runpath;
	    
	    range_active = false;  
	  } else 
	    util_abort("%s: something wrong when parsing: \"%s\" to integer \n",__func__ , token_list[token_index]);
	}
	token_index++;
      }
    } while (token_index < tokens);
  }
}



static void enkf_config_set_data_file(enkf_config_type * config , const char * data_file) {
  if (util_file_exists(data_file))
    config->data_file = util_realloc_string_copy(config->data_file , data_file);
  else 
    util_abort("%s: sorry: data_file:%s does not exist - aborting \n",__func__ , data_file);
}


/**
  This function sets the schedule files. The schedule_src_file is
  general file (absolut or relative pointing anywhere), which point to
  an existing SCHEDULE file.

  The target file is the name of the SCHEDULE file the EnKF
  application generates at run-time. This MUST just be a name, not a
  full path (not checked). If the target file argument is NULL, the
  basename of the src_file is used.

*/

static void enkf_config_set_schedule_files(enkf_config_type * config , const char * schedule_src_file , const char * schedule_target_file) { 
  if (util_file_exists(schedule_src_file))
    config->schedule_src_file = util_realloc_string_copy(config->schedule_src_file , schedule_src_file); 
  else
    util_abort("%s: sorry:schedule_src_file:%s does not exist - aborting \n",__func__ ,schedule_src_file); 

  if (schedule_target_file != NULL) 
    config->schedule_target_file = util_realloc_string_copy(config->schedule_target_file , schedule_target_file); 
  else {
    char * base_name;
    char * extension;

    util_alloc_file_components(schedule_src_file , NULL , &base_name , &extension);
    if (extension != NULL) {
      config->schedule_target_file = util_malloc(strlen( base_name) + strlen( extension) + 2 , __func__);
      sprintf(config->schedule_target_file , "%s.%s" , base_name , extension);
      free(base_name);
      free(extension);
    } else 
      config->schedule_target_file = base_name;
  }

}



const char * enkf_config_get_schedule_src_file(const enkf_config_type * config) {
  return config->schedule_src_file;
}

const char * enkf_config_get_schedule_target_file(const enkf_config_type * config) {
  return config->schedule_target_file;
}


const char * enkf_config_get_obs_config_file(const enkf_config_type * config) {
  return config->obs_config_file;
}


static void enkf_config_set_grid(enkf_config_type * config, const char * grid_file) {
  if (config->grid == NULL) 
    config->grid = ecl_grid_alloc(grid_file , config->endian_swap);
  else
    util_exit("%s: sorry - can only have *ONE* GRID occurence in config file. \n",__func__);
}



static void enkf_config_set_ens_size(enkf_config_type * config , int ens_size) {
  if (ens_size > 0) {
    if (config->ens_size < 0) {
      int iens;
      config->ens_size  = ens_size;
      config->keep_runpath = util_malloc(ens_size * sizeof * config->keep_runpath , __func__);
      for (iens = 0; iens < ens_size; iens++)
	config->keep_runpath[iens] = false; /* Default is *NOT* to store anything */

    } else
      util_exit("%s: sorry - can only set ensemble size *ONE* time in the config file. \n",__func__);
  } else 
    util_abort("%s: size must be greater than zero - aborting \n",__func__);
}


/*
   1 may 1999
*/

static void enkf_config_set_start_date(enkf_config_type * config , const char ** tokens) {
  int day , year;
  bool ok = true;

  ok = util_sscanf_int(tokens[0] , &day);
  ok = ok && util_sscanf_int(tokens[2] , &year); 
  if (ok) {
    int month_nr = util_get_month_nr(tokens[1]);
    config->start_time = util_make_date( day , month_nr , year );
  } else 
    util_abort("%s: fatal error when parsing start_time: \"%s %s %s\" \n",__func__, tokens[0] , tokens[1] , tokens[2]);
}



/**
   This function adds a keyword to the list of restart keywords wich
   are included. Observe that ecl_util_escape_kw() is called prior to
   adding it.

   The kw __ALL__ is magic; and will result in a request to store all
   static kewyords. This wastes disk-space, but might be beneficial
   when debugging.
*/

void enkf_config_add_static_kw(enkf_config_type * enkf_config , const char * _kw) {
  if (strcmp(_kw , "__ALL__") == 0) 
    enkf_config->include_all_static_kw = true;
  else {
    char * kw = util_alloc_string_copy(_kw);
    ecl_util_escape_kw(kw);
    set_add_key(enkf_config->static_kw_set , kw);
    free(kw);
  }
}


/**
   This function checks whether the static kw should be
   included. Observe that it is __assumed__ that ecl_util_escape_kw()
   has already been called on the kw.
*/

bool enkf_config_include_static_kw(const enkf_config_type * enkf_config , const char * kw) {
  if (enkf_config->include_all_static_kw)
    return true;
  else
    return set_has_key(enkf_config->static_kw_set , kw);
}



static enkf_config_type * enkf_config_alloc_empty(bool fmt_file ,
						  bool unified  ,         
						  bool endian_swap) {
  enkf_config_type * config = malloc(sizeof * config);
  config->config_hash = hash_alloc();
  config->data_kw     = hash_alloc();

  config->endian_swap   = endian_swap;
  config->unified       = unified;
  config->fmt_file      = fmt_file;

  /*
    all of these must be set before the config object is ready for use.
  */
  config->eclbase   	       = NULL;
  config->data_file 	       = NULL;
  config->grid      	       = NULL;
  config->run_path  	       = NULL;
  config->schedule_src_file    = NULL;
  config->schedule_target_file = NULL;
  config->start_time           = -1;
  config->ens_size             = -1;
  config->keep_runpath         = NULL;
  config->obs_config_file      = NULL;
  config->ens_path             = NULL;
  config->result_path          = NULL;
  config->init_file            = NULL;
  config->forward_model        = NULL;
  config->enkf_sched_file      = NULL;
  enkf_config_set_debug(config , 0);

  config->include_all_static_kw = false;
  config->static_kw_set         = set_alloc_empty();
  
  enkf_config_add_static_kw(config , "INTEHEAD");
  enkf_config_add_static_kw(config , "LOGIHEAD"); 
  enkf_config_add_static_kw(config , "DOUBHEAD"); 
  enkf_config_add_static_kw(config , "IGRP"); 
  enkf_config_add_static_kw(config , "SGRP"); 
  enkf_config_add_static_kw(config , "XGRP"); 
  enkf_config_add_static_kw(config , "ZGRP"); 
  enkf_config_add_static_kw(config , "IWEL"); 
  enkf_config_add_static_kw(config , "SWEL"); 
  enkf_config_add_static_kw(config , "XWEL"); 
  enkf_config_add_static_kw(config , "ZWEL"); 
  enkf_config_add_static_kw(config , "ICON"); 
  enkf_config_add_static_kw(config , "SCON"); 
  enkf_config_add_static_kw(config , "XCON"); 
  enkf_config_add_static_kw(config , "HIDDEN"); 
  enkf_config_add_static_kw(config , "STARTSOL"); 
  enkf_config_add_static_kw(config , "PRESSURE"); 
  enkf_config_add_static_kw(config , "SWAT"); 
  enkf_config_add_static_kw(config , "SGAS"); 
  enkf_config_add_static_kw(config , "RS"); 
  enkf_config_add_static_kw(config , "RV"); 
  enkf_config_add_static_kw(config , "ENDSOL"); 

  enkf_config_add_static_kw(config , "ICAQNUM");
  enkf_config_add_static_kw(config , "IAAQ");
  enkf_config_add_static_kw(config , "ICAQ");
  enkf_config_add_static_kw(config , "SCAQNUM");
  enkf_config_add_static_kw(config , "SAAQ");
  enkf_config_add_static_kw(config , "SCAQ");
  enkf_config_add_static_kw(config , "ACAQNUM");
  enkf_config_add_static_kw(config , "XAAQ");
  enkf_config_add_static_kw(config , "ACAQ");

  enkf_config_add_static_kw(config , "ISEG");
  enkf_config_add_static_kw(config , "ILBS");
  enkf_config_add_static_kw(config , "ILBR");
  enkf_config_add_static_kw(config , "RSEG");

  enkf_config_add_static_kw(config , "ISTHW");
  enkf_config_add_static_kw(config , "ISTHG");


  return config;
}



void enkf_config_add_type(enkf_config_type * enkf_config , 
			  const char    * key      , 
			  enkf_var_type enkf_type  , 
			  enkf_impl_type impl_type , 
			  const char   * enkf_outfile  , 
			  const char   * enkf_infile ,  
			  const void   * data) {

  if (enkf_config_has_key(enkf_config , key)) 
    util_abort("%s: a configuration object:%s has already been added - aborting \n",__func__ , key);
  
  {
    config_free_ftype * freef = NULL;
    switch(impl_type) {
    case(FIELD):
      freef             = field_config_free__;
      break;
    case(MULTZ):
      freef             = multz_config_free__;
      break;
    case(RELPERM):
      freef             = relperm_config_free__;
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
      freef             = NULL; 
      break;
    case(GEN_KW):
      freef             = gen_kw_config_free__;
      break;
    case(SUMMARY):
      freef             = summary_config_free__;
      break;
    case(HAVANA_FAULT):
      freef             = havana_fault_config_free__;
      break;
    case(GEN_DATA):
      freef             = gen_data_config_free__;
      break;
    case(GEN_PARAM):
      freef             = gen_param_config_free__;
      break;
    default:
      util_abort("%s : invalid implementation type: %d - aborting \n",__func__ , impl_type);
    }
    
    {
      enkf_config_node_type * node = enkf_config_node_alloc(enkf_type , impl_type , key , enkf_outfile , enkf_infile , data , freef);
      hash_insert_hash_owned_ref(enkf_config->config_hash , key , node , enkf_config_node_free__);
    }
  }
}


void enkf_config_add_well(enkf_config_type * enkf_config , const char *well_name , int size, const char ** var_list) {
  enkf_config_add_type(enkf_config , well_name , ecl_summary , WELL , NULL , NULL , well_config_alloc(well_name , size , var_list));
}


void enkf_config_add_gen_kw(enkf_config_type * enkf_config , const char * config_file) {
  enkf_config_add_type(enkf_config , "gen_kw" , parameter , GEN_KW , NULL , NULL , gen_kw_config_fscanf_alloc(config_file , NULL));
}







const stringlist_type * enkf_config_get_forward_model(const enkf_config_type * config) {
  return config->forward_model;
}

void enkf_config_set_forward_model(enkf_config_type * config , const char  ** forward_model , int forward_model_length) {
  if (config->forward_model != NULL)
    stringlist_free(config->forward_model);

  config->forward_model = stringlist_alloc_argv_copy(forward_model , forward_model_length);
}





#define __assert_not_null(p , t , OK) if (p == NULL ) { fprintf(stderr,"The key:%s must be set.\n",t); OK = false; }
static void enkf_config_post_assert(const enkf_config_type * config , ext_joblist_type * joblist) {
  bool OK = true;
  
  __assert_not_null(config->forward_model      , "FORWARD_MODEL" , OK);
  __assert_not_null(config->ens_path           , "ENSPATH"  	 , OK);
  __assert_not_null(config->schedule_src_file  , "SCHEDULE" 	 , OK);
  __assert_not_null(config->schedule_src_file  , "SCHEDULE" 	 , OK);
  __assert_not_null(config->result_path        , "RESULT_PATH"  , OK);

  if (config->ens_size <= 0) {
    fprintf(stderr,"Must set ensemble size > 0 with KEYWORD SIZE.\n");
    OK = false;
  }

  if (OK) {
    int i;
    for (i=0; i < stringlist_get_argc(config->forward_model); i++) {
      if ( !ext_joblist_has_job(joblist , stringlist_iget(config->forward_model , i))) {
	OK = false;
	fprintf(stderr, "%s: FORWARD_MODEL:%s not defined \n",__func__ , stringlist_iget(config->forward_model , i));
      }
    }
  }
  if (!OK) util_exit("Exiting due to errors in config file\n");
}
#undef __assert_not_null



#define ASSERT_TOKENS(kw,t,n) if ((t - 1) < (n)) util_abort("%s: when parsing %s must have at least %d arguments - aborting \n",__func__ , kw , (n)); 
enkf_config_type * enkf_config_fscanf_alloc(const char * __config_file , 
					    enkf_site_config_type * site_config , 
					    ext_joblist_type * joblist , 
					    bool fmt_file ,
					    bool unified  ,         
					    bool endian_swap) { 
  char *config_file;
  {
    char * path;
    char * base;
    char * ext;
    util_alloc_file_components(__config_file , &path , &base , &ext);
    if (path != NULL) {
      if (chdir(path) != 0) 
	util_abort("%s: failed to change directory to: %s : %s \n",__func__ , path , strerror(errno));

      printf("Changing to directory ...................: %s \n",path);
      if (ext != NULL) {
	config_file = util_alloc_joined_string((const char *[3]) {base , "." , ext} , 3 , "");
	free(base);
      } else 
	config_file = base;
      free(ext);
      free(path);
    } else
      config_file = util_alloc_string_copy(__config_file);
  }
  
  printf("Loading configuration information from ..: %s \n",config_file);  
  {  
    enkf_config_type * enkf_config = enkf_config_alloc_empty(fmt_file , unified , endian_swap);
    FILE * stream = util_fopen(config_file , "r");
    char * line;
    bool at_eof = false;

    
    do {
      enkf_impl_type impl_type;
      int i , tokens;
      int active_tokens;
      char **token_list;
      bool debug = (enkf_config_get_debug(enkf_config) > 0);
      
      line  = util_fscanf_alloc_line(stream , &at_eof);
      if (line != NULL) {
	util_split_string(line , " \t" , &tokens , &token_list);
      
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
	  if (debug) printf("Parsing: %s \n",token_list[0]);
	  impl_type = enkf_types_check_impl_type(token_list[0]);
	  if (impl_type == INVALID) {
	    const char * kw = token_list[0];
	    if (enkf_site_config_has_key(site_config , kw)) {
	      /* The configuration overrides a value from the site_config object. */
	      ASSERT_TOKENS(kw , active_tokens , 1);
	      enkf_site_config_set_argv(site_config , kw ,  active_tokens - 1 , (const char **) &token_list[1] );
	    } else {
	      if (strcmp(kw , "ADD_STATIC_KW") == 0) {
		int ikw;
		for (ikw = 1; ikw < active_tokens; ikw++)
		  enkf_config_add_static_kw(enkf_config , token_list[ikw]);
	      } else if (strcmp(kw , "SIZE") == 0) {
		ASSERT_TOKENS("SIZE" , active_tokens , 1);
		{
		  int ens_size;
		  if (util_sscanf_int(token_list[1] , &ens_size)) 
		    enkf_config_set_ens_size( enkf_config , ens_size);
		  else 
		    util_abort("%s: failed to convert:%s to valid integer - aborting \n",__func__ , token_list[1]);
		}
	      } else if (strcmp(kw , "OBS_CONFIG") == 0) {
		ASSERT_TOKENS("OBS_CONFIG" , active_tokens , 1);
		{
		  char * obs_config_file = token_list[1];
		  enkf_config_set_obs_config_file(enkf_config , obs_config_file);
		}
	      } else if (strcmp(kw , "ENSPATH") == 0) {
		ASSERT_TOKENS("ENSPATH" , active_tokens , 1);
		enkf_config_set_ens_path(enkf_config , token_list[1]);
	      } else if (strcmp(kw , "EQUIL_INIT_FILE") == 0) {
		ASSERT_TOKENS("EQUIL_INIT_FILE" , active_tokens , 1);
		enkf_config_set_init_file(enkf_config , token_list[1]);
	      } else if (strcmp(kw , "DATA_KW") == 0) {
		/* This is later taken over by the enkf_state objects */
		ASSERT_TOKENS("DATA_KW" , active_tokens , 2);
		hash_insert_hash_owned_ref(enkf_config->data_kw , token_list[1] , util_alloc_joined_string((const char **) &token_list[2] , active_tokens - 2 , " ") , free);
	      } else if (strcmp(kw , "DEBUG") == 0) {
		{
		  int debug_level;
		  ASSERT_TOKENS("DEBUG_LEVEL" , active_tokens , 1);
		  
		  if (util_sscanf_int(token_list[1] , &debug_level))
		      enkf_config_set_debug(enkf_config ,debug_level);
		  else
		    util_abort("%s: failed to parse debug_level:%s to an integer \n",__func__ , token_list[1]);
		  
		}
	      } else if (strcmp(kw , "RUNPATH") == 0) {
		ASSERT_TOKENS("RUNPATH" , active_tokens , 1);
		enkf_config_set_run_path( enkf_config , token_list[1] );
	      } else if (strcmp(kw, "ENKF_SCHED_FILE") == 0) {
		ASSERT_TOKENS("ENKF_SCHED_FILE" , active_tokens , 1);
		enkf_config_set_enkf_sched_file(enkf_config , token_list[1]);
	      } else if (strcmp(kw , "DATA_FILE") == 0) {
		ASSERT_TOKENS("DATA_FILE" , active_tokens , 1);
		enkf_config_set_data_file( enkf_config , token_list[1] );
	      } else if (strcmp(kw , "ECLBASE") == 0) {
		ASSERT_TOKENS("ECLBASE" , active_tokens , 1);
		enkf_config_set_eclbase( enkf_config , token_list[1] );
	      } else if (strcmp(kw , "RESULT_PATH") == 0) {
		ASSERT_TOKENS("RESULT_PATH" , active_tokens , 1);
		enkf_config_set_result_path( enkf_config , token_list[1] );
	      } else if (strcmp(kw , "SCHEDULE_FILE") == 0) {
		ASSERT_TOKENS("SCHEDULE_FILE" , active_tokens , 1);
		if (active_tokens == 3)
		  enkf_config_set_schedule_files(enkf_config , token_list[1] , token_list[2]);
		else 
		  enkf_config_set_schedule_files(enkf_config , token_list[1] , NULL);
	      } else if (strcmp(kw , "KEEP_RUNPATH") == 0) {
		ASSERT_TOKENS("KEEP_RUNPATH" , active_tokens , 2);
		enkf_config_set_ecl_store(enkf_config , true , active_tokens - 1 , (const char **) &token_list[1]);
	      } else if (strcmp(kw , "GRID") == 0) {
		ASSERT_TOKENS("GRID" , active_tokens , 1);
		enkf_config_set_grid(enkf_config , token_list[1]);
	      } else if (strcmp(kw , "START_TIME") == 0) {
		ASSERT_TOKENS("START_TIME" , active_tokens , 3);
		enkf_config_set_start_date(enkf_config , (const char **) &token_list[1]);
	      } else if (strcmp(kw , "FORWARD_MODEL") == 0) {
		ASSERT_TOKENS("FORWARD_MODEL" , active_tokens , 1);
		enkf_config_set_forward_model(enkf_config , (const char **) &token_list[1] , active_tokens - 1);
	      } else
		fprintf(stderr,"%s: ** Warning ** keyword: %s not recognzied - line ignored \n",__func__ , kw);
	    }    
	  } else {
	    switch(impl_type) {
	    case(GEN_PARAM):
	      ASSERT_TOKENS("GEN_PARAM" , active_tokens , 3);
	      {
		const char * key         = token_list[1];
		const char * ecl_file    = token_list[2];
		const char * init_fmt    = token_list[3];
		char       * ecl_template;   

		if (active_tokens == 5)
		  ecl_template = token_list[4];
		else
		  ecl_template = NULL;
		
		enkf_config_add_type(enkf_config , key , parameter , GEN_PARAM , ecl_file , NULL , gen_param_config_alloc( init_fmt , ecl_template ));
	      }
	      break;
	    case(MULTZ):
	      ASSERT_TOKENS("MULTZ" , active_tokens , 3);
	      {
		const char * key         = token_list[1];
		const char * ecl_file    = token_list[2];
		char       * config_file = token_list[3];
		int   nx,ny,nz,active_size;
		
		if (enkf_config->grid == NULL) 
		  util_abort("%s must add grid prior to adding MULTZ - aborting \n",__func__);

		ecl_grid_get_dims(enkf_config->grid , &nx , &ny , &nz , &active_size);
		enkf_config_add_type(enkf_config , key , parameter , MULTZ , ecl_file , NULL , multz_config_fscanf_alloc(config_file , nx , ny , nz));
	      }
	      break;
	    case(RELPERM):
	      {
		const char * key         = token_list[1];
		const char * ecl_file    = token_list[2];
		char       * config_file = token_list[3];
		char       * table_file  = token_list[4];
		enkf_config_add_type(enkf_config, key,parameter, RELPERM, ecl_file, NULL , relperm_config_fscanf_alloc(config_file,table_file));
	      }
	      break;
	    case(MULTFLT):
	      ASSERT_TOKENS("MULTFLT" , active_tokens , 3);
	      {
		const char * key         = token_list[1];
		const char * ecl_file    = token_list[2];
		char       * config_file = token_list[3];
		enkf_config_add_type(enkf_config , key , parameter , MULTFLT , ecl_file , NULL , multflt_config_fscanf_alloc(config_file));
	      }
	      break;
	    case(HAVANA_FAULT):

	      {
		const char * key         = token_list[1];
		const char * config_file = token_list[2];
		enkf_config_add_type(enkf_config , key , parameter , HAVANA_FAULT , NULL , NULL , havana_fault_config_fscanf_alloc(config_file));
	      }
	      break;
	    case(EQUIL):
	      ASSERT_TOKENS("EQUIL" , active_tokens , 3);
	      {
		const char * key         = token_list[1];
		const char * ecl_file    = token_list[2];
		char       * config_file = token_list[3];
		enkf_config_add_type(enkf_config , key , parameter , EQUIL , ecl_file , NULL , equil_config_fscanf_alloc(config_file));
	      }
	      break;
	    case(FIELD):
	      ASSERT_TOKENS("FIELD" , active_tokens , 2);
	      {
		const char * key             = token_list[1];
		const char * var_type_string = token_list[2];
		int   nx,ny,nz,active_size;
		
		if (enkf_config->grid == NULL) 
		  util_abort("%s must add grid prior to adding FIELD - aborting \n",__func__);
		
		ecl_grid_get_dims(enkf_config->grid , &nx , &ny , &nz , &active_size);
		if (strcmp(var_type_string , "DYNAMIC") == 0)
		  enkf_config_add_type(enkf_config , key , ecl_restart , FIELD , NULL , NULL , field_config_alloc_dynamic(key , enkf_config->grid));
		else if (strcmp(var_type_string , "PARAMETER") == 0) {
		  ASSERT_TOKENS("FIELD" , active_tokens , 5);
		  {
		    const char * ecl_file = token_list[3];
		    int init_mode;
		    if (util_sscanf_int(token_list[4] , &init_mode)) 
		      enkf_config_add_type(enkf_config , key , parameter   , FIELD , ecl_file , NULL , field_config_alloc_parameter(key , 
															     enkf_config->grid,
															     0 , init_mode , active_tokens - 5 , (const char **) &token_list[5]));
		    else 
		      util_abort("%s: init mode must be valid int - aborting \n",__func__);
		  }
		} else if (strcmp(var_type_string , "GENERAL") == 0) {
		  const char * enkf_outfile = token_list[3];
		  const char * enkf_infile  = token_list[4];
		  const char * init_fmt     = token_list[5];
		  enkf_config_add_type(enkf_config , key , ecl_restart , FIELD , enkf_outfile , enkf_infile , field_config_alloc_general(key , enkf_config->grid , init_fmt));
		} else util_abort("%s: -- \n",__func__);
	      }
	      break;
	    case(WELL):
	      ASSERT_TOKENS("WELL" , active_tokens , 2);
	      enkf_config_add_well(enkf_config , token_list[1] , active_tokens - 2 , (const char **) &token_list[2]);
	      break;
	    case(SUMMARY):
	      ASSERT_TOKENS("SUMMARY" , active_tokens , 2);
	      enkf_config_add_type(enkf_config , token_list[1] , ecl_summary , SUMMARY , NULL , NULL , summary_config_alloc(active_tokens - 2 , (const char **) &token_list[2]));
	      break;
	    case(GEN_DATA):
	      enkf_config_add_type(enkf_config , token_list[1] , ecl_restart , GEN_DATA , NULL , NULL , gen_data_config_fscanf_alloc(token_list[2]));
	      break;
	    case(GEN_KW):
	      ASSERT_TOKENS("GEN_KW" , active_tokens , 4);
	      {
		const char * key              = token_list[1];
		const char * template_file    = token_list[2];
		const char * target_file      = token_list[3];
		const char * config_file      = token_list[4];
		enkf_config_add_type(enkf_config , key , parameter , GEN_KW , target_file , NULL , gen_kw_config_fscanf_alloc(config_file , template_file));
	      }
	      break;
	    default:
	      util_abort("%s: Invalid keyword: %s - aborting \n",__func__ ,  token_list[0]);
	    }
	  }
	}
	util_free_stringlist(token_list , tokens);
	free(line);
      }
    } while (!at_eof);
    free(config_file);
    enkf_site_config_finalize(site_config , joblist);
    enkf_config_post_assert(enkf_config , joblist);
    
    enkf_site_config_validate(site_config);
    return enkf_config;
  }	
}
#undef ASSERT_TOKENS






const enkf_config_node_type * enkf_config_get_node_ref(const enkf_config_type * ens, const char * key) {
  if (hash_has_key(ens->config_hash , key)) {
    enkf_config_node_type * node = hash_get(ens->config_hash , key);
    return node;
  } else {
    util_abort("%s: ens node:%s does not exist \n",__func__ , key);
    return NULL; /* Compiler shut up */
  }
}


path_fmt_type * enkf_config_get_run_path_fmt( const enkf_config_type * config) { return config->run_path; }
path_fmt_type * enkf_config_get_eclbase_fmt ( const enkf_config_type * config) { return config->eclbase; }

char * enkf_config_alloc_result_path(const enkf_config_type * config , int report_step) {
  return path_fmt_alloc_path(config->result_path , true , report_step);
}


int    	       enkf_config_get_ens_size  (const enkf_config_type * config) 	     { return config->ens_size; }
time_t 	       enkf_config_get_start_date(const enkf_config_type * config) 	     { return config->start_time; }
bool           enkf_config_iget_ecl_store(const enkf_config_type * config, int iens) { return config->keep_runpath[iens]; }



void enkf_config_free(enkf_config_type * config) {  
  hash_free(config->config_hash);
  path_fmt_free(config->result_path);
  path_fmt_free(config->run_path);
  path_fmt_free(config->eclbase);
  ecl_grid_free(config->grid);
  free(config->data_file);
  util_safe_free(config->obs_config_file);
  util_safe_free(config->enkf_sched_file);
  if (config->ens_path != NULL) free(config->ens_path);
  if (config->schedule_src_file != NULL) {
    free(config->schedule_src_file);
    free(config->schedule_target_file);
  }
  hash_free(config->data_kw);
  free(config->init_file);
  free(config->keep_runpath);
  stringlist_free(config->forward_model);
  set_free(config->static_kw_set);
  free(config);
}




job_queue_type * enkf_config_alloc_job_queue(const enkf_config_type * config , const enkf_site_config_type * site_config) {
  job_queue_type          * job_queue;
  int                       max_running  = 0;  
  basic_queue_driver_type * queue_driver = NULL;

  const char * queue_system = enkf_site_config_get_value(site_config , "QUEUE_SYSTEM");
  if (strcmp(queue_system , "LSF") == 0) {
    stringlist_type * resource_request_list = enkf_site_config_alloc_argv(site_config , "LSF_RESOURCES");
    const char * queue_name                 = enkf_site_config_get_value(site_config , "LSF_QUEUE");
    max_running  = strtol(enkf_site_config_get_value(site_config , "MAX_RUNNING_LSF") , NULL , 10);
    queue_driver = lsf_driver_alloc(queue_name , resource_request_list);
    stringlist_free(resource_request_list);
  } else if (strcmp(queue_system , "LOCAL") == 0) {
    queue_driver = local_driver_alloc();
    max_running  = strtol(enkf_site_config_get_value(site_config , "MAX_RUNNING_LOCAL") , NULL , 10);
  } else if (strcmp(queue_system , "RSH") == 0) {
    {
      stringlist_type * host_list  = enkf_site_config_alloc_argv(site_config , "RSH_HOST_LIST");
      queue_driver = rsh_driver_alloc(enkf_site_config_get_value(site_config , "RSH_COMMAND") , host_list);
      stringlist_free(host_list);
    }
    max_running  = strtol(enkf_site_config_get_value(site_config , "MAX_RUNNING_RSH") , NULL , 10);
  } else 
    util_abort("%s: internal error - queue_system:%s not recognized - aborting \n",__func__ , queue_system);

  {
    int max_submit  = 2;
    job_queue = job_queue_alloc(enkf_config_get_ens_size(config),
				max_running , 
				max_submit  ,
				enkf_site_config_get_value(site_config , "JOB_SCRIPT"),
				queue_driver);
    
  }
  return job_queue;
}


