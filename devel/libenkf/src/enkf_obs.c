/**
   See the file README.obs for ducumentation of the varios datatypes
   involved with observations/measurement/+++.
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <hash.h>
#include <util.h>
#include <enkf_obs.h>
#include <ecl_rft_node.h>
#include <well_obs.h>
#include <ensemble_config.h>
#include <obs_node.h>
#include <history.h>
#include <enkf_util.h>
#include <sched_file.h>
#include <summary_obs.h>
#include <gen_obs.h>
#include <gen_data_config.h>

static int enkf_obs_sscanf_report_step(const enkf_obs_type * enkf_obs , const char * meas_time_string) {
  int report_step;

  if (!util_sscanf_int(meas_time_string , &report_step)) {
    time_t meas_time;
    if ( util_sscanf_date(meas_time_string , &meas_time)) 
      report_step = sched_file_get_restart_nr_from_time_t(enkf_obs->sched_file , meas_time);
    else
      util_abort("%s: failed to parse: \"%s\" as a date (Format: DD-MM-YYYY) or report_step.\n",__func__ , meas_time_string);
  }

  return report_step;
}


static time_t enkf_obs_sscanf_obs_time(const enkf_obs_type * enkf_obs , const char * meas_time_string) {
  time_t meas_time;
  if (!util_sscanf_date(meas_time_string , &meas_time)) {
    int report_step;
    if (!util_sscanf_int(meas_time_string , &report_step)) 
      meas_time = sched_file_iget_block_end_time(enkf_obs->sched_file , report_step); /* _end_time / _start_time ?? */
    else
      util_abort("%s: failed to parse: \"%s\" as a date (Format: DD-MM-YYYY) or report_step.\n",__func__ , meas_time_string);
  }
  return meas_time;
}




static enkf_obs_type * enkf_obs_alloc(const sched_file_type * sched_file , const history_type * hist) {
  enkf_obs_type * enkf_obs = malloc(sizeof * enkf_obs);
  enkf_obs->obs_hash 	   = hash_alloc(10);
  
  
  enkf_obs->sched_file     = sched_file;
  enkf_obs->hist     	   = hist;
  enkf_obs->num_reports    = sched_file_get_num_restart_files(sched_file); /*history_get_num_reports(enkf_obs->hist);*/
  return enkf_obs;
}







/*
  Observe that the keywords used to index into the hash_obs() structure
  must be the same as is used to get the enkf_node object from the enkf_state.
*/
  
void enkf_obs_add_obs(enkf_obs_type * enkf_obs, const char * key , const obs_node_type * node) {
  if (hash_has_key(enkf_obs->obs_hash , key)) {
    fprintf(stderr,"%s: observation with key:%s already added - aborting \n",__func__ , key);
    abort();
  }
  hash_insert_hash_owned_ref(enkf_obs->obs_hash , key , node , obs_node_free__);
}




/* const void * __enkf_obs_ens_get(const enkf_obs_type * enkf_obs , const char * key , enkf_impl_type impl_type) { */
/*   if (enkf_ens_has_key(enkf_obs->ens , key)) { */
/*     const enkf_config_node_type * node = enkf_ens_get_config_ref(enkf_obs->ens , key ); */
/*     if (enkf_config_node_get_impl_type(node) == impl_type)  */
/*       return enkf_config_node_get_ref(node);       */
/*     else { */
/*       fprintf(stderr,"%s: obse node:%s is not of correct type - aborting \n",__func__ , key); */
/*       abort(); */
/*     } */
/*   } else { */
/*     fprintf(stderr,"%s: can not find ens key for observation:%s - aborting \n",__func__ , key); */
/*     abort(); */
/*   } */
/* } */



/* enkf_obs_type * enkf_obs_fscanf_alloc(const char * filename , const enkf_ens_type * ens , const history_type * hist) { */
/*   const int num_reports = 100; */
/*   enkf_obs_type * enkf_obs = enkf_obs_alloc(ens , hist , num_reports); */
/*   FILE * stream = enkf_util_fopen_r(filename , __func__); */
/*   char *path; */
  
/*   util_alloc_file_components(filename , &path , NULL , NULL); */
/*   do { */
/*     char key[64]; */
/*     char obs_type[64]; */
/*     int scan_count = fscanf(stream , "%s  %s" , key , obs_type); */
/*     if (scan_count != EOF) { */
/*       char file[64]; */
/*       time_t obs_time = -1; */
/*       int    active   = -1; */
/*       char   cmp; */
/*       obs_node_type * obs_node; */

/*       if (scan_count != 2) { */
/* 	fprintf(stderr,"%s: fatal error when loading: %s - aborting \n",__func__ , filename); */
/* 	abort(); */
/*       } */
/*       if (fscanf(stream , "%d" , &active) != 1) { */
/* 	fscanf(stream , "%c" , &cmp); */
/* 	obs_time = util_fscanf_date(stream); */
/*       } */
/*       if (fscanf(stream ,"%s" , file) == 1) { */
/* 	/\* */
/* 	  Now we have scanned the full line sucessfully ...  */
/* 	*\/ */
/* 	char * ens_file = util_alloc_full_path(path , file); */
/* 	enkf_active_type     active_mode; */
/* 	const void * ens; */
/* 	void       * obs; */

/* 	if (active == 1)  */
/* 	  active_mode = active_on; */
/* 	else if (active == 0) */
/* 	  active_mode = active_off; */
/* 	else if (active == -1) { */
/* 	  switch(cmp) { */
/* 	  case('='): */
/* 	    active_mode = active_at; */
/* 	    break;  */
/* 	  case('>'): */
/* 	    active_mode = active_after; */
/* 	    break; */
/* 	  case('<'): */
/* 	    active_mode = active_before; */
/* 	    break; */
/* 	  default: */
/* 	    fprintf(stderr,"%s: comparison operator:%c in file:%s not recognized - alternatives are: = < > \n",__func__ , cmp , filename); */
/* 	    abort(); */
/* 	  } */
/* 	} else { */
/* 	  fprintf(stderr,"%s: variable active:%d is invalid - aborting \n",__func__ , active); */
/* 	  abort(); */
/* 	} */

/* 	if (strcmp(obs_type , WELL_OBS_TYPE_STRING) == 0) {       */
/* 	  ens = __enkf_obs_ens_get(enkf_obs , key , WELL); */
/* 	  obs = well_obs_fscanf_alloc(ens_file , ens , hist); */
/* 	} else if (strcmp(obs_type , POINT_OBS_TYPE_STRING) == 0) { */
/* 	  ens = __enkf_obs_ens_get(enkf_obs , key , FIELD); */
/* 	  obs = /\*well_obs_fscanf_alloc(well_ens , hist);*\/ NULL; */
/* 	} else { */
/* 	  fprintf(stderr,"%s: observation type: %s is not recognized - aborting \n",__func__ , obs_type); */
/* 	  abort(); */
/* 	} */
	
/* 	obs_node = obs_node_alloc(obs , active_mode , obs_time , well_obs_get_observations__ , well_obs_measure__ , well_obs_free__); */
/* 	enkf_obs_add_obs(enkf_obs , key , obs_node); */
/* 	free(ens_file); */
/*       } */
/*     } */
/*   } while ( !feof(stream) ); */
  
/*   free(path); */
/*   fclose(stream); */
/*   return enkf_obs; */
/* } */



void enkf_obs_free(enkf_obs_type * enkf_obs) {
  hash_free(enkf_obs->obs_hash);
  free(enkf_obs);
}


void enkf_obs_add_gen_obs(enkf_obs_type * enkf_obs , const char * key , enkf_config_node_type * config_node ) {
  gen_obs_type * gen_obs = gen_obs_alloc( enkf_config_node_get_ref( config_node ));
  enkf_obs_add_obs(enkf_obs , key , obs_node_alloc(gen_obs , key , key , enkf_obs->num_reports , true , gen_obs_get_observations__ , gen_obs_measure__ , gen_obs_free__));
}


/*
  Observations should probably have a name of some sort 
*/

void enkf_obs_add_well_obs(enkf_obs_type * enkf_obs, const enkf_config_node_type * config_node , const char * well_name , const char * obs_label , const char * config_file) {
  bool default_active = true;
  well_obs_type * well_obs = well_obs_fscanf_alloc(config_file , enkf_config_node_get_ref(config_node) , enkf_obs->hist , enkf_obs->sched_file);
  enkf_obs_add_obs(enkf_obs , well_name , obs_node_alloc(well_obs , well_name , obs_label , enkf_obs->num_reports , default_active , well_obs_get_observations__ , well_obs_measure__ , well_obs_free__));
}



/**
   This function adds a summary observation. 
*/




void enkf_obs_add_summary_obs_from_file(enkf_obs_type * enkf_obs, const enkf_config_node_type * config_node , const char * state_kw , const char * var , const char * data_file) {
  const bool default_active = false;
  summary_obs_type * summary_obs;
  obs_node_type    * obs_node;
  int              * report_nr;
  int                i , size;

  {
    double  * value;
    double  * std;
    char   ** meas_time_string;
  
    summary_obs_fscanf_alloc_data(data_file , &size , &meas_time_string , &value , &std);
    report_nr = util_malloc(size * sizeof * report_nr , __func__);
    for (i=0; i < size; i++)
      report_nr[i] = enkf_obs_sscanf_report_step(enkf_obs , meas_time_string[i]);
    
    summary_obs = summary_obs_alloc(enkf_config_node_get_ref(config_node) , var , size , report_nr , value , std);
    free(std);
    free(value);
  
    util_free_stringlist(meas_time_string , size);
  }
  obs_node = obs_node_alloc(summary_obs , state_kw , var , enkf_obs->num_reports , default_active , summary_obs_get_observations__ , summary_obs_measure__ , summary_obs_free__);
  for (i=0;  i < size; i++)
    obs_node_activate_report_step(obs_node , report_nr[i] , report_nr[i]);
  
  enkf_obs_add_obs(enkf_obs , var , obs_node);
  free(report_nr);
}


void enkf_obs_add_summary_obs_from_history(enkf_obs_type * enkf_obs, const enkf_config_node_type * config_node, const char * state_kw, const char * var, const history_type * history, bool is_group_kw, const char * config_file)
{
  summary_obs_type * summary_obs;
  obs_node_type    * obs_node;
  int              * report_nr;
  int                size;
  util_abort("%s: Not implemented.\n", __func__);
}



/**
  This functions loads block data observations. The return values are
  given by references. The format of the file should be as follows:

  i1  j1   k1   value1  std1
  i2  j2   k2   value2  std2
  i3  j3   k3   value3  std3
  ...
  ...

  Blank lines are *NOT* allowed. 
*/


static void enkf_obs_fscanf_alloc_block_data(const char * filename , int * _size , int ** _i , int ** _j , int ** _k , double ** _data , double ** _std) {
  int size;
  int *i , *j , *k;
  double * data , * std;

  FILE * stream = util_fopen(filename , "r");
  size = util_count_file_lines(stream);
  if (size > 0) {
    i    = util_malloc(size * sizeof * i    , __func__);
    j    = util_malloc(size * sizeof * j    , __func__);
    k    = util_malloc(size * sizeof * k    , __func__);
    data = util_malloc(size * sizeof * data , __func__);
    std  = util_malloc(size * sizeof * std  , __func__);

    {
      int line_nr;
      for (line_nr = 0; line_nr < size; line_nr++) {
	if (fscanf(stream , "%d %d %d %lg %lg" , &i[line_nr] , &j[line_nr] , &k[line_nr] , &data[line_nr] , &std[line_nr]) != 5) {
	  char * line;
	  bool   at_eof;
	  util_rewind_line(stream);
	  line = util_fscanf_alloc_line(stream , &at_eof);
	  util_abort("%s: something failed when reading line %d: \"%s\" in %s. Expected format: \"i  j  k   value   std\".\n",__func__ , line_nr + 1 , line , filename);
	}
      }
    }
  
    *_i 	 = i;
    *_j 	 = j;
    *_k 	 = k;
    *_data = data;
    *_std  = std;
  }
  *_size = size;
  fclose(stream);
}


static void enkf_obs_add_field_obs__(enkf_obs_type * enkf_obs, const enkf_config_node_type * config_node , const char * ecl_field, const char * obs_label , int size, const int *i , const int *j , const int *k, const double * obs_data , const double * obs_std , time_t meas_time) {
  bool default_active        		    = false;
  field_obs_type * field_obs 		    = field_obs_alloc(enkf_config_node_get_ref(config_node) , ecl_field , size , i , j , k , obs_data , obs_std);
  obs_node_type  * obs_node  		    = obs_node_alloc(field_obs , ecl_field , obs_label , enkf_obs->num_reports , default_active , field_obs_get_observations__ , field_obs_measure__ , field_obs_free__);

  if (meas_time != -1)
    obs_node_activate_time_t(obs_node , enkf_obs->sched_file , meas_time , meas_time);
  enkf_obs_add_obs(enkf_obs , obs_label , obs_node);
}



void enkf_obs_add_rft_obs(enkf_obs_type * enkf_obs , const enkf_config_node_type * config_node , const ecl_rft_node_type * rft_node, const double * p_data , const double * p_std) {
  char * obs_label = util_alloc_string_sum2("RFT/" , ecl_rft_node_well_name_ref(rft_node));
  enkf_obs_add_field_obs__(enkf_obs , config_node , "PRESSURE" ,  obs_label , ecl_rft_node_get_size(rft_node) , ecl_rft_node_get_i(rft_node), ecl_rft_node_get_j(rft_node), ecl_rft_node_get_k(rft_node) , p_data , p_std , ecl_rft_node_get_recording_time(rft_node));
  free(obs_label);
}


void enkf_obs_add_field_obs(enkf_obs_type * enkf_obs, const enkf_config_node_type * config_node , const char * block_file , const char * ecl_field, const char * obs_label , time_t meas_time) {
  int *i;
  int *j;
  int *k;
  int size;
  double * data , * std;
  

  enkf_obs_fscanf_alloc_block_data(block_file , &size , &i , &j , &k , &data , &std);
  if (size > 0) {
    enkf_obs_add_field_obs__(enkf_obs , config_node , ecl_field , obs_label , size , i , j , k , data , std , meas_time);
    free(i);
    free(j);
    free(k);
    free(data);
    free(std);
  }

}




void enkf_obs_get_observations(enkf_obs_type * enkf_obs , int report_step , obs_data_type * obs_data) {
  char ** obs_keys = hash_alloc_keylist(enkf_obs->obs_hash);
  int iobs;

  obs_data_reset(obs_data);
  for (iobs = 0; iobs < hash_get_size(enkf_obs->obs_hash); iobs++) {
    obs_node_type * obs_node = hash_get(enkf_obs->obs_hash , obs_keys[iobs]);
    obs_node_get_observations(obs_node , report_step , obs_data);
  }
  util_free_stringlist( obs_keys , hash_get_size(enkf_obs->obs_hash));
  
}

  


#define ASSERT_TOKENS(kw,t,n) if ((t - 1) < (n)) { fprintf(stderr,"%s: when parsing %s must have at least %d arguments - aborting \n",__func__ , kw , (n)); abort(); }
enkf_obs_type * enkf_obs_fscanf_alloc(const char * config_file, const ensemble_config_type * ensemble_config , const sched_file_type * sched_file , const history_type * hist) {
  enkf_obs_type * enkf_obs = enkf_obs_alloc(sched_file , hist);
  if (config_file != NULL) {
    FILE * stream = util_fopen(config_file , "r");
    bool   at_eof;
    do {
      int active_tokens , tokens;
      char  *line;
      char **token_list;
      
      line  = util_fscanf_alloc_line(stream , &at_eof);
      if (line != NULL) {
	int i;
	util_split_string(line , " " , &tokens , &token_list);
	
	active_tokens = tokens;
	for (i = 0; i < tokens; i++) {
	  if (token_list[i][0] == '-') {
	    if (token_list[i][1] == '-') {
	      active_tokens = i;
	      break;
	    }	  }
	}
	if (active_tokens > 0) {
	  const char *kw           = token_list[0];
	  if (strcmp(kw , "WELL") == 0) {
	    ASSERT_TOKENS("WELL" , active_tokens , 2);
	    {
	      const char * well_name   = token_list[1];
	      char * config_file = token_list[2];
	      char * obs_label   = NULL;
	      const enkf_config_node_type * config_node = ensemble_config_get_node(ensemble_config , well_name);
	      enkf_obs_add_well_obs(enkf_obs , config_node , well_name , obs_label , config_file);
	    }
	  }
	  else if (strcmp(kw , "BLOCK") == 0) {
	    ASSERT_TOKENS("BLOCK" , active_tokens , 4);
	    {
	      const char * obs_label 	    = token_list[1];
	      const char * field     	    = token_list[2];
	      const char * meas_time_string = token_list[3];
	      const char * data_file        = token_list[4];
	      const enkf_config_node_type * config_node = ensemble_config_get_node(ensemble_config , field);
	      time_t meas_time = enkf_obs_sscanf_obs_time(enkf_obs , meas_time_string);
	      enkf_obs_add_field_obs(enkf_obs , config_node , data_file , field , obs_label , meas_time);
	    }
	  } else if (strcmp(kw , "SUMMARY") == 0) {
	    ASSERT_TOKENS("SUMMARY" , active_tokens , 3);
	    /*
	      SUMMARY RPR RPR:2 data_file

	      RPR:    keyword in enkf_state lookup
	      RPR:2:  summary variable to look up - this will be the unique obs_key
	    */
	    
	    const char * state_kw         = token_list[1];
	    const char * var              = token_list[2];
	    const char * data_file        = token_list[3];
	    const enkf_config_node_type * config_node = ensemble_config_get_node(ensemble_config , state_kw);
	    enkf_obs_add_summary_obs_from_file(enkf_obs , config_node , state_kw , var , data_file);
	  } else if (strcmp(kw , "GEN_OBS") == 0) {
	    ASSERT_TOKENS("GEN_OBS" , active_tokens , 1);
	    const char * state_kw         = token_list[1];
	    enkf_config_node_type * config_node = ensemble_config_get_node(ensemble_config , state_kw);
	    enkf_obs_add_gen_obs(enkf_obs , state_kw , config_node);
	  } else
	    fprintf(stderr," ** Warning ** keyword:%s not recognized when parsing: %s - ignored \n",kw , config_file);
	  
	}
	util_free_stringlist(token_list , tokens);
	free(line);
      }
    } while ( !at_eof );
    fclose(stream);
  }
  return enkf_obs;
}

bool enkf_obs_get_local_active(ensemble_config_type * ensemble_config, int report_step){
  bool local_active;
  const gen_data_config_type * gen_data_config = enkf_obs_get_gen_data_config(ensemble_config);
  local_active = gen_data_config_get_local_active(gen_data_config, report_step);
  return local_active;
}

/* This function returns a gen_data_config object */
gen_data_config_type * enkf_obs_get_gen_data_config(ensemble_config_type * ensemble_config){
  const char * state_gen_data = "AI";
  const enkf_config_node_type * config_node_gen_data = ensemble_config_get_node(ensemble_config , state_gen_data);
  gen_data_config_type * gen_data_config = enkf_config_node_get_ref(config_node_gen_data);  
  return gen_data_config;
}

void enkf_obs_change_gen_data_config_iactive(ensemble_config_type * ensemble_config, int local_step){
  gen_data_config_type * gen_data_config = enkf_obs_get_gen_data_config(ensemble_config);
  gen_data_config_change_iactive(gen_data_config,local_step);
  
}


int enkf_obs_get_num_local_updates(ensemble_config_type * ensemble_config){
  int num_local_updates;
  gen_data_config_type * gen_data_config = enkf_obs_get_gen_data_config(ensemble_config);
  num_local_updates = gen_data_config_get_num_local_updates(gen_data_config);
  return num_local_updates;
}

void enkf_obs_set_local_step(ensemble_config_type * ensemble_config, int local_step){
  gen_data_config_type * gen_data_config = enkf_obs_get_gen_data_config(ensemble_config);
  gen_data_config_set_local_step(gen_data_config,local_step);
}

