#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <enkf_util.h>
#include <well_obs.h>
#include <hash.h>
#include <meas_op.h>
#include <meas_vector.h>
#include <hash.h>
#include <list.h>
#include <history.h>
#include <sched_file.h>
#include <well_config.h>
#include <well.h>



typedef struct obs_error_struct      obs_error_type;
typedef struct well_var_obs_struct   well_var_obs_type;


struct obs_error_struct {
  int size;
  double   	      * abs_std;
  double   	      * rel_std; 
  bool                * active;
  enkf_obs_error_type * error_mode;
};



struct well_var_obs_struct {
  char                   * var;
  bool                     currently_active;        
  obs_error_type         * error;
};


struct well_obs_struct {
  const well_config_type * config;
  const history_type     * hist;
  const sched_file_type  * sched_file;
  hash_type              * var_hash;
  int                      __num_reports;
};


/*****************************************************************/



static void obs_error_realloc(obs_error_type * well_error, int new_size) {
  int old_size           = well_error->size;
  well_error->abs_std    = enkf_util_realloc(well_error->abs_std    , new_size * sizeof * well_error->abs_std    , __func__);
  well_error->rel_std    = enkf_util_realloc(well_error->rel_std    , new_size * sizeof * well_error->rel_std    , __func__);
  well_error->error_mode = enkf_util_realloc(well_error->error_mode , new_size * sizeof * well_error->error_mode , __func__);
  well_error->active     = enkf_util_realloc(well_error->active     , new_size * sizeof * well_error->active     , __func__);
  well_error->size       = new_size;

  if (new_size > old_size && old_size > 0) {
    int report_nr;
    for (report_nr = old_size; report_nr < new_size; report_nr++) {
      well_error->abs_std[report_nr]    = well_error->abs_std[old_size-1];
      well_error->rel_std[report_nr]    = well_error->rel_std[old_size-1];
      well_error->error_mode[report_nr] = well_error->error_mode[old_size-1];
      well_error->active[report_nr]     = well_error->active[old_size-1];
    }
  }
}




static obs_error_type * obs_error_alloc(int size) {
  obs_error_type * well_error = malloc(sizeof * well_error);

  well_error->size       = 0;
  well_error->abs_std    = NULL;
  well_error->rel_std    = NULL;
  well_error->error_mode = NULL;
  well_error->active     = NULL;
  obs_error_realloc(well_error , size);
  
  return well_error;
}




static void obs_error_free(obs_error_type * well_error) {
  free(well_error->abs_std);
  free(well_error->rel_std);
  free(well_error->error_mode);
  free(well_error->active);
  free(well_error);
}


static void obs_error_iset(obs_error_type * well_error , int report , double abs_std , double rel_std , enkf_obs_error_type error_mode , bool active) {
  if (report < 0 || report >= well_error->size) {
    fprintf(stderr,"%s report_nr:%d not in interval [0,%d> - aborting \n",__func__ , report , well_error->size);
    abort();
  }
  {
    switch (error_mode) {
    case(abs_error):
      break;
    case(rel_error):
      break;
    case(rel_min_abs_error):
      break;
    default:
      fprintf(stderr,"%s: internal error: error_mode:%d invalid - aborting \n",__func__ , error_mode);
      abort();
    }
  }
  well_error->abs_std[report]    = abs_std;
  well_error->rel_std[report]    = rel_std;
  well_error->error_mode[report] = error_mode;
  well_error->active[report]     = active;
}



static void obs_error_set_block(obs_error_type * well_error , int first_report , int last_report , double abs_std , double rel_std , enkf_obs_error_type error_mode , bool active) {
  int report_nr;
  for (report_nr = 0; report_nr <= last_report; report_nr++)
    obs_error_iset(well_error , report_nr , abs_std , rel_std , error_mode , active);
}


static void obs_error_set_all(obs_error_type * well_error , double abs_std , double rel_std , enkf_obs_error_type error_mode , bool active) {
  obs_error_set_block(well_error , 0 , well_error->size - 1, abs_std , rel_std , error_mode , active);
}

static void obs_error_set_last(obs_error_type * well_error , int report_size, double abs_std , double rel_std , enkf_obs_error_type error_mode , bool active) {
  obs_error_set_block(well_error , well_error->size - report_size , well_error->size - 1, abs_std , rel_std , error_mode , active);
}

static void obs_error_set_first(obs_error_type * well_error , int report_size, double abs_std , double rel_std , enkf_obs_error_type error_mode , bool active) {
  obs_error_set_block(well_error , 0 , report_size , abs_std , rel_std , error_mode , active);
}

static double obs_error_iget_std(obs_error_type * well_error , int report_step, double data) {
  if (report_step < 0 || report_step >= well_error->size) {
    fprintf(stderr,"%s report_nr:%d not in interval [0,%d> - aborting \n",__func__ , report_step , well_error->size);
    abort();
  }
  {
    double std;
    switch (well_error->error_mode[report_step]) {
    case(abs_error):
      std = well_error->abs_std[report_step];
      break;
    case(rel_error):
      std = well_error->rel_std[report_step] * data;
      break;
    case(rel_min_abs_error):
      std = util_double_min( well_error->rel_std[report_step] * data , well_error->abs_std[report_step]);
      break;
    default:
      fprintf(stderr,"%s: internal error: error_mode:%d invalid - aborting \n",__func__ , well_error->error_mode[report_step]);
      abort();
    }
    return std;
  }
}


static bool obs_error_iactive(const obs_error_type * obs , int report_step) {
  if (report_step < 0 || report_step >= obs->size) {
    fprintf(stderr,"%s report_nr:%d not in interval [0,%d> - aborting \n",__func__ , report_step , obs->size);
    abort();
  }
  return obs->active[report_step];
}

/*****************************************************************/

well_var_obs_type * well_var_obs_alloc(const char * var , int size) {
  well_var_obs_type * well_var_obs = util_malloc(sizeof * well_var_obs , __func__);

  well_var_obs->var   		 = util_alloc_string_copy(var);
  well_var_obs->error 		 = obs_error_alloc(size);
  well_var_obs->currently_active = false;
  obs_error_set_block(well_var_obs->error , 0 , size - 1 , 0.0 , 0.0 , abs_error , false);
  
  return well_var_obs;
}


void well_var_obs_free(well_var_obs_type * well_var_obs) {
  obs_error_free(well_var_obs->error);
  free(well_var_obs->var);
  free(well_var_obs);
}


void well_var_obs_free__(void * well_var_obs) {
  well_var_obs_free( (well_var_obs_type *) well_var_obs );
}



/*****************************************************************/

static well_obs_type * __well_obs_alloc(const well_config_type * config , const history_type * hist , const sched_file_type * sched_file) {

  well_obs_type * well_obs   = malloc(sizeof * well_obs);
  well_obs->hist             = hist;
  well_obs->sched_file       = sched_file;
  well_obs->config           = config;
  well_obs->var_hash         = hash_alloc(10);
  well_obs->__num_reports    = history_get_num_reports(hist);
  return well_obs;
}



static void well_obs_add_var(const well_obs_type * well_obs , const char * var) {
  if (well_config_has_var(well_obs->config , var)) {
    well_var_obs_type * well_var = well_var_obs_alloc(var , well_obs->__num_reports);
    hash_insert_hash_owned_ref(well_obs->var_hash , var , well_var , well_var_obs_free__);
  } else {
    fprintf(stderr,"%s: well: %s does not have variable:%s - aborting \n",__func__ , well_config_get_well_name_ref(well_obs->config) , var);
    abort();
  }
}


static bool well_obs_has_var(const well_obs_type * well_obs , const char * var) {
  return hash_has_key(well_obs->var_hash , var);
}


static well_var_obs_type * well_obs_get_var(const well_obs_type * well_obs , const char * var) {
  return hash_get(well_obs->var_hash , var);
}


int well_obs_parse_report_nr(const well_obs_type * well_obs , const char * token , int default_value) {
  int report;
  if (token[0] == '*')
    report = default_value;
  else {
    time_t date;
    if (util_sscanf_date(token , &date)) {
      report = sched_file_time_t_to_report_step(well_obs->sched_file , date );
      if (report == -1) abort();
    } else { 
      if (!util_sscanf_int(token , &report)) {
	fprintf(stderr,"%s: failed to parse:\"%s\" to report_nr or date - aborting \n",__func__ , token);
	abort();
      }
    }
  }
  return report;
}



/*
WOPR (*|10|01/01/2002) - (*|12|01/01/2003) (ON|OFF) (ABS|REL|RELMIN) std1  (std2)
*/
/*#define ASSERT_TOKENS(kw,t,n) if ((t - 1) < (n)) { fprintf(stderr,"%s: when parsing %s must have at least %d arguments - aborting \n",__func__ , kw , (n)); abort(); }*/
well_obs_type * well_obs_fscanf_alloc(const char * filename , const well_config_type * config , const history_type * hist , const sched_file_type * sched_file) {
  FILE * stream = enkf_util_fopen_r(filename , __func__);
  well_obs_type * well_obs = __well_obs_alloc(config , hist , sched_file);
  bool at_eof;
  do {
    char  * line  = util_fscanf_alloc_line(stream , &at_eof);
    char ** token_list;
    int     i , tokens , active_tokens;
    
    if (line != NULL) {
      char * var;
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
      
      if (active_tokens >= 5) {
	well_var_obs_type * well_var;
	var = token_list[0];
	if ( !well_obs_has_var(well_obs , var) )
	  well_obs_add_var(well_obs , var);

	well_var = well_obs_get_var(well_obs , var);
	{
	  char *on_off = token_list[4];
	  int report1 , report2;
	  report1 = well_obs_parse_report_nr(well_obs , token_list[1] , 0);
	  if ( ! (token_list[2][0] == '-') ) {
	    fprintf(stderr,"%s: expected \"-\" separating dates|report_nr - aborting\n",__func__);
	    abort();
	  }
	  report2 = well_obs_parse_report_nr(well_obs , token_list[3] , well_obs->__num_reports - 1);
	  if (strcmp(on_off , "OFF") == 0) 
	    obs_error_set_block(well_var->error , report1 , report2 , 0.0 , 0.0 , abs_error , false);
	  else if (strcmp(on_off , "ON") == 0) {
	    if (active_tokens >= 7) {
	      double std1 , std2;
	      char * error_mode = token_list[5];
	      if (sscanf(token_list[6] , "%lg" , &std1) != 1) {
		fprintf(stderr,"%s: failed to parse: %s as floating number - aborting \n",__func__ , token_list[6]);
		abort();
	      }
	      if (strcmp(error_mode , "ABS") == 0) 
		obs_error_set_block(well_var->error , report1 , report2 , std1 , 0.0  , abs_error , true);
	      else if (strcmp(error_mode , "REL") == 0)
		obs_error_set_block(well_var->error , report1 , report2 , 0.0  , std1 , rel_error , true);
	      else if (strcmp(error_mode , "RELMIN") == 0) {
		if (active_tokens >= 8) {
		  if (sscanf(token_list[7] , "%lg" , &std2) == 1)
		    obs_error_set_block(well_var->error , report1 , report2 , std1 , std2 , rel_min_abs_error , true);
		  else {
		    fprintf(stderr,"%s: could not parse: %s as floating number - aborting \n",__func__ , token_list[7]);
		    abort();
		  }
		} else {
		  fprintf(stderr,"%s: to few tokens - aborting \n",__func__);
		  abort();
		}
	      }
	    }  
	  } else {
	    fprintf(stderr,"%s: did not recognize:%s - aborting \n",__func__ ,on_off);
	    abort();
	  }
	}
      } else {
	if (active_tokens > 0)
	  fprintf(stderr,"%s ** Warning ** line:%s ignored \n",__func__ , line);
      }
      util_free_string_list(token_list , tokens);
    }
    free(line);
  } while (!at_eof);
  fclose(stream);
  return well_obs;
}






static double well_obs_get_observation__(const history_type * hist , int report_step , const char * well_name , const char * var, bool *active) {
  bool default_used;
  double d = history_get2(hist , report_step , well_name , var , &default_used);
  if (default_used || (d == 0.0))
    *active = false;
  else
    *active = true;
  return d;
}



void well_obs_get_observations(const well_obs_type * well_obs , int report_step, obs_data_type * obs_data) {
  const char *well_name = well_config_get_well_name_ref(well_obs->config);
  char ** var_list;
  const int kw_len = 16;
  char kw[kw_len+1];
  int i;
  var_list = hash_alloc_keylist(well_obs->var_hash);
  for (i = 0; i < hash_get_size(well_obs->var_hash); i++) {
    well_var_obs_type * var = well_obs_get_var(well_obs , var_list[i]);
    var->currently_active = false;
    if (obs_error_iactive(var->error , report_step)) {
      double d   = well_obs_get_observation__(well_obs->hist , report_step , well_name , var_list[i] , &var->currently_active);
      if (var->currently_active) {
	double std = obs_error_iget_std(var->error , report_step , d);
	strncpy(kw , well_name   , kw_len);
	strcat(kw , "/");
	strncat(kw , var_list[i] , kw_len - 1 - (strlen(well_name)));
	obs_data_add(obs_data , d , std , kw);
      } 
    }
  }
  hash_free_ext_keylist(well_obs->var_hash , var_list);
}



void well_obs_measure(const well_obs_type * well_obs , const well_type * well_state , meas_vector_type * meas_vector) {
  int i;
  char ** var_list;
  var_list = hash_alloc_keylist(well_obs->var_hash);
  
  for (i=0; i < hash_get_size(well_obs->var_hash); i++) {
    well_var_obs_type * obs = well_obs_get_var(well_obs , var_list[i]);
    if (obs->currently_active) 
      meas_vector_add(meas_vector , well_get(well_state , var_list[i]));
  }
  hash_free_ext_keylist(well_obs->var_hash , var_list);
}


void well_obs_free(well_obs_type * well_obs) {
  hash_free(well_obs->var_hash);
  free(well_obs);
}





VOID_FREE(well_obs)
VOID_GET_OBS(well_obs)
VOID_MEASURE(well)
