#include <stdlib.h>
#include <util.h>
#include <stdio.h>
#include <summary_obs.h>
#include <obs_data.h>
#include <meas_matrix.h>
#include <summary.h>



struct summary_obs_struct {
  const summary_config_type * config;
  char   		    * var_string;
  int                         size;
  double                     *value;
  double 		     *std;         
  int                        *report_nr; 
};





/** 
This function is called from enkf_obs.c. It used to parse a file with the following format:

Date|report_nr value std

10-10-2005     267.0  10.0
10             278.0   5.0
.....

*/

void summary_obs_fscanf_alloc_data(const char * data_file , int * _size , char *** _meas_time_string , double ** _data , double ** _std) {
  int total_size , size;
  char  ** meas_time_string;
  double * data;
  double * std;
  bool     at_eof = false;
  FILE * stream   = util_fopen(data_file , "r");
  
  total_size        = util_count_file_lines(stream);
  meas_time_string = util_malloc(total_size * sizeof * meas_time_string , __func__);
  data             = util_malloc(total_size * sizeof * data             , __func__);
  std              = util_malloc(total_size * sizeof * std              , __func__);
  size = 0;
  
  while ( !at_eof ) {
    char * line = util_fscanf_alloc_line(stream , &at_eof);
    if (!at_eof && (line != NULL)) {
      int    tokens;
      char **token_list;
      util_split_string(line , " " , &tokens , &token_list);
      if (tokens > 0) {
	if (tokens != 3) 
	  util_abort("%s: something wrong with line:\'%s\' when parsing:%s \n",__func__ , line , data_file);
      
	meas_time_string[size] = util_alloc_string_copy(token_list[0]);
	if ( !util_sscanf_double( token_list[1] , &data[size])) util_abort("%s: could not parse:%s into a double - aborting", __func__ , token_list[1]);
	if ( !util_sscanf_double( token_list[2] , &std[size]))  util_abort("%s: could not parse:%s into a double - aborting", __func__ , token_list[2]);
	size++;

	util_free_stringlist( token_list , tokens );
      }
    }
    free(line);
  }
  
  if (size != total_size)
    meas_time_string = util_realloc(meas_time_string , size * sizeof * meas_time_string , __func__);
  
  *_meas_time_string = meas_time_string;
  *_data             = data;
  *_std              = std;
  *_size             = size;
}


void summary_obs_fprintf(const summary_obs_type * summary_obs , FILE * stream) {
  int istep;
  fprintf(stream, "Observing summary variable....: %s \n",summary_obs->var_string);
  for (istep = 0; istep < summary_obs->size; istep++)
    fprintf(stream , "   Report step: %3d     Observation: %8.3f +/- %8.3f \n",summary_obs->report_nr[istep] , summary_obs->value[istep] , summary_obs->std[istep]);
}



/**
  This function allocates a summary_obs instance. The var string
  should be of the format WOPR:OP_4 used by the summary.x
  program. Observe that this format is *not* checked before the actual
  observation time.

  The report_nr, value and std vectors are typically allocated with
  the function summary_obs_fscanf_alloc_data.
*/

summary_obs_type * summary_obs_alloc(const summary_config_type * config , const char * var_string , int size, const int * report_nr , const double * value , const double * std) {
  summary_obs_type * obs = util_malloc(sizeof * obs , __func__);
  
  obs->config     = config;
  obs->var_string = util_alloc_string_copy(var_string);
  obs->size       = size;
  obs->report_nr  = util_alloc_copy(report_nr , size * sizeof * report_nr , __func__);
  obs->value  	  = util_alloc_copy(value     , size * sizeof * value 	  , __func__);
  obs->std    	  = util_alloc_copy(std       , size * sizeof * std   	  , __func__);

  return obs;
}



void summary_obs_free(summary_obs_type * obs) {
  free(obs->var_string);
  free(obs->report_nr);
  free(obs->value);
  free(obs->std);
  free(obs);
}



void summary_obs_get_observations(const summary_obs_type * summary_obs , int report_step, obs_data_type * obs_data) {
  int ir;
  int report_index = -1;

  for (ir = 0; ir < summary_obs->size; ir++) {
    if (summary_obs->report_nr[ir] == report_step) {
      report_index = ir;
      break;
    }
  }

  if ( report_index == -1 ) 
    util_abort("%s: could not find summary observation for report_step:%d.\n",__func__ , report_step);
  
  obs_data_add(obs_data , summary_obs->value[report_index] , summary_obs->std[report_index] , summary_obs->var_string);
}



void summary_obs_measure(const summary_obs_type * obs , const summary_type * summary , meas_vector_type * meas_vector) {
  meas_vector_add(meas_vector , summary_get(summary , obs->var_string));
}




VOID_FREE(summary_obs)
VOID_GET_OBS(summary_obs)
VOID_MEASURE(summary)
