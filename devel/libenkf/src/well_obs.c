#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <enkf_util.h>
#include <well_obs.h>
#include <meas_op.h>
#include <meas_vector.h>
#include <hash.h>
#include <list.h>
#include <history.h>
#include <well_config.h>
#include <well.h>



typedef struct well_obs_error_struct well_obs_error_type;


struct well_obs_error_struct {
  int size;
  double   	    * abs_std;
  double   	    * rel_std; 
  enkf_obs_err_type * error_mode;
};


struct well_obs_struct {
  const well_config_type * config;
  int    	           size;
  const history_type     *  hist;
  char   	       	 ** var_list;

  bool                   *  currently_active;        
  well_obs_error_type    ** error;
};


/*****************************************************************/



static void well_obs_error_realloc(well_obs_error_type * well_error, int new_size) {
  int old_size           = well_error->size;
  well_error->abs_std    = enkf_util_realloc(well_error->abs_std    , new_size * sizeof * well_error->abs_std    , __func__);
  well_error->rel_std    = enkf_util_realloc(well_error->rel_std    , new_size * sizeof * well_error->rel_std    , __func__);
  well_error->error_mode = enkf_util_realloc(well_error->error_mode , new_size * sizeof * well_error->error_mode , __func__);
  well_error->size       = new_size;

  if (new_size > old_size && old_size > 0) {
    int report_nr;
    for (report_nr = old_size; report_nr < new_size; report_nr++) {
      well_error->abs_std[report_nr]    = well_error->abs_std[old_size-1];
      well_error->rel_std[report_nr]    = well_error->rel_std[old_size-1];
      well_error->error_mode[report_nr] = well_error->error_mode[old_size-1];
    }
  }
}




static well_obs_error_type * well_obs_error_alloc(int size) {
  well_obs_error_type * well_error = malloc(sizeof * well_error);

  well_error->size       = 0;
  well_error->abs_std    = NULL;
  well_error->rel_std    = NULL;
  well_error->error_mode = NULL;
  well_obs_error_realloc(well_error , size);
  
  return well_error;
}




static void well_obs_error_free(well_obs_error_type * well_error) {
  free(well_error->abs_std);
  free(well_error->rel_std);
  free(well_error->error_mode);
  free(well_error);
}


static void well_obs_error_iset(well_obs_error_type * well_error , int report , double abs_std , double rel_std , enkf_obs_error_type error_mode) {
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
}



static void well_obs_error_set_block(well_obs_error_type * well_error , int first_report , int last_report , double abs_std , double rel_std , enkf_obs_error_type error_mode) {
  int report_nr;
  for (report_nr = 0; report_nr <= last_report; report_nr++)
    well_obs_error_iset(well_error , report_nr , abs_std , rel_std , error_mode);
}


static void well_obs_error_set_all(well_obs_error_type * well_error , double abs_std , double rel_std , enkf_obs_error_type error_mode) {
  well_obs_error_set_block(well_error , 0 , well_error->size - 1, abs_std , rel_std , error_mode);
}

static void well_obs_error_set_last(well_obs_error_type * well_error , int report_size, double abs_std , double rel_std , enkf_obs_error_type error_mode) {
  well_obs_error_set_block(well_error , well_error->size - report_size , well_error->size - 1, abs_std , rel_std , error_mode);
}

static void well_obs_error_set_first(well_obs_error_type * well_error , int report_size, double abs_std , double rel_std , enkf_obs_error_type error_mode) {
  well_obs_error_set_block(well_error , 0 , report_size , abs_std , rel_std , error_mode);
}

static double well_obs_error_iget_std(well_obs_error_type * well_error , int report_step, double data) {
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


/*****************************************************************/

static well_obs_type * __well_obs_alloc(const well_config_type * config , int size, int report_size , const history_type * hist) {

  well_obs_type * well_obs   = malloc(sizeof * well_obs);
  well_obs->size      	     = size;
  well_obs->var_list   	     = enkf_util_malloc(size * sizeof * well_obs->var_list   , __func__);
  well_obs->error            = enkf_util_malloc(size * sizeof * well_obs->error      , __func__);
  {
    int ivar;
    for (ivar = 0; ivar < size; ivar++)
      well_obs->error[ivar] = well_obs_error_alloc(report_size);
  }
  well_obs->currently_active = enkf_util_malloc(size * sizeof * well_obs->currently_active , __func__);
  well_obs->hist             = hist;
  well_obs->config           = config;

  return well_obs;
}



well_obs_type * well_obs_fscanf_alloc(const char * filename , const well_config_type * config , const history_type * hist) {
  FILE * stream = enkf_util_fopen_r(filename , __func__);
  int size      = util_count_file_lines(stream);
  well_obs_type * well_obs = __well_obs_alloc(config , size , history_get_num_reports(hist) , hist);
  char * line = NULL;
  bool at_eof;
  int ivar;

  fseek(stream , 0L , SEEK_SET);
  for (ivar = 0; ivar < size; ivar++) {
    line = util_fscanf_realloc_line(stream , &at_eof , line);
    if (!at_eof) {
      double abs_std, rel_std;
      int error_mode;
      char var[32];
      int scan_count = sscanf(line , "%s    %d  %lg  %lg" , var, &error_mode , &abs_std , &rel_std);
      if (scan_count == 4) {
	if (well_config_has_var(config , var)) {
	  well_obs->var_list[ivar]   = util_alloc_string_copy(var);
	  well_obs_error_set_all(well_obs->error[ivar] , abs_std , rel_std , error_mode);
	} else {
	  fprintf(stderr,"%s: attempt to observe %s/%s but the variable:%s is not added to the state vector - aborting.\n",__func__ , well_config_get_well_name_ref(config) , var , var);
	  abort();
	}
      } else {
	fprintf(stderr,"%s: invalid input line %d :\"%s\" - aborting \n",__func__ , ivar + 1, line);
	abort();
      }
    }
  } 

  free(line);
  fclose(stream);
  return well_obs;
}





well_obs_type * well_obs_alloc(const well_config_type * config , int nvar , const char ** var_list , const history_type * hist , const double * abs_std , const double * rel_std , const enkf_obs_error_type * error_mode ) {
  int i;
  well_obs_type * well_obs  = __well_obs_alloc(config , nvar , history_get_num_reports(hist) , hist);

  for (i=0; i < nvar; i++) {
    if (well_config_has_var(config , var_list[i])) 
      well_obs->var_list[i] = util_alloc_string_copy(var_list[i]);
    else {
      fprintf(stderr,"%s: attempt to observe %s/%s but the variable:%s is not added to the state vector - aborting.\n",
	      __func__ , well_config_get_well_name_ref(config) , var_list[i] , var_list[i]);
      abort();
    }
    well_obs_error_set_all(well_obs->error[i] , abs_std[i] , rel_std[i] , error_mode[i]);
  }
  return well_obs;
}




void well_obs_get_observations(const well_obs_type * well_obs , int report_step, obs_data_type * obs_data) {
  const char *well_name = well_config_get_well_name_ref(well_obs->config);
  const int kw_len = 16;
  char kw[kw_len+1];
  int i;
  for (i=0; i < well_obs->size; i++) {
    bool default_used;
    double d   = history_get2(well_obs->hist , report_step , well_name , well_obs->var_list[i] , &default_used);
    if (!default_used) {
      double std = well_obs_error_iget_std(well_obs->error[i] , report_step , d);
      strncpy(kw , well_name   , kw_len);
      strcat(kw , "/");
      strncat(kw , well_obs->var_list[i] , kw_len - 1 - (strlen(well_name)));
      obs_data_add(obs_data , d , std , kw);
      well_obs->currently_active[i] = true;
    } else 
      well_obs->currently_active[i] = false;
  }
}



void well_obs_measure(const well_obs_type * well_obs , const well_type * well_state , meas_vector_type * meas_vector) {
  int i;

  for (i=0; i < well_obs->size; i++) 
    if (well_obs->currently_active[i])
      meas_vector_add(meas_vector , well_get(well_state , well_obs->var_list[i]));
  
}


void well_obs_free(well_obs_type * well_obs) {
  util_free_string_list(well_obs->var_list , well_obs->size);
  free(well_obs->currently_active);
  {
    int ivar;
    for (ivar = 0; ivar < well_obs->size; ivar++)
      well_obs_error_free(well_obs->error[ivar]);
  }
  free(well_obs);
}




VOID_FREE(well_obs)
VOID_GET_OBS(well_obs)
VOID_MEASURE(well)
