#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <hash.h>
#include <util.h>
#include <enkf_obs.h>
#include <ecl_rft_node.h>
#include <well_obs.h>
#include <obs_node.h>
#include <history.h>
#include <enkf_util.h>
#include <sched_file.h>
#include <enkf_config.h>







static enkf_obs_type * enkf_obs_alloc(const sched_file_type * sched_file , const history_type * hist) {
  enkf_obs_type * enkf_obs = malloc(sizeof * enkf_obs);
  enkf_obs->obs_hash 	   = hash_alloc();
  
  
  enkf_obs->sched_file     = sched_file;
  enkf_obs->hist     	   = hist;
  enkf_obs->num_reports    = history_get_num_reports(enkf_obs->hist);
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


/*
  Observations should probably have a name of some sort 
*/

void enkf_obs_add_well_obs(enkf_obs_type * enkf_obs, const enkf_config_node_type * config_node , const char * well_name , const char * obs_label , const char * config_file) {
  bool default_active = true;
  well_obs_type * well_obs = well_obs_fscanf_alloc(config_file , enkf_config_node_get_ref(config_node) , enkf_obs->hist , enkf_obs->sched_file);
  enkf_obs_add_obs(enkf_obs , well_name , obs_node_alloc(well_obs , obs_label , enkf_obs->num_reports , default_active , well_obs_get_observations__ , well_obs_measure__ , well_obs_free__));
}






void enkf_obs_add_field_obs(enkf_obs_type * enkf_obs, const enkf_config_node_type * config_node , const char * ecl_field, const char * obs_label , int size, const int *i , const int *j , const int *k, const double * obs_data , time_t meas_time) {
  bool default_active = false;

  field_obs_type * field_obs = field_obs_alloc(enkf_config_node_get_ref(config_node) , ecl_field , size , i , j , k , obs_data);
  obs_node_type  * obs_node  = obs_node_alloc(field_obs , obs_label , enkf_obs->num_reports , default_active , field_obs_get_observations__ , field_obs_measure__ , field_obs_free__);
  if (meas_time != -1)
    obs_node_activate_time_t(obs_node , enkf_obs->sched_file , meas_time , meas_time);
  enkf_obs_add_obs(enkf_obs , ecl_field , obs_node);
}




void enkf_obs_add_rft_obs(enkf_obs_type * enkf_obs , const enkf_config_node_type * config_node , const ecl_rft_node_type * rft_node, const double * p_data) {
  char * obs_label = util_alloc_string_sum2("RFT/" , ecl_rft_node_well_name_ref(rft_node));
  enkf_obs_add_field_obs(enkf_obs , config_node , "PRES" ,  obs_label , ecl_rft_node_get_size(rft_node) , ecl_rft_node_get_i(rft_node), ecl_rft_node_get_j(rft_node), ecl_rft_node_get_k(rft_node) , p_data , ecl_rft_node_get_recording_time(rft_node));
  free(obs_label);
}




void enkf_obs_get_observations(enkf_obs_type * enkf_obs , int report_step , obs_data_type * obs_data) {
  bool valid;
  obs_node_type * obs_node = hash_iter_get_first(enkf_obs->obs_hash , &valid);
  while (valid) {
    obs_node_get_observations(obs_node , report_step , obs_data);
    obs_node = hash_iter_get_next(enkf_obs->obs_hash , &valid);
  }
}



#define ASSERT_TOKENS(kw,t,n) if ((t - 1) < (n)) { fprintf(stderr,"%s: when parsing %s must have at least %d arguments - aborting \n",__func__ , kw , (n)); abort(); }
enkf_obs_type * enkf_obs_fscanf_alloc(const enkf_config_type * config , const sched_file_type * sched_file , const history_type * hist) {
  enkf_obs_type * enkf_obs = enkf_obs_alloc(sched_file , hist);
  const char * config_file = enkf_config_get_obs_config_file(config);
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
	    }
	  }
	}
	if (active_tokens > 0) {
	  const char *kw           = token_list[0];
	  const char * well_name   = token_list[1];
	  char * config_file = token_list[2];
	  char * obs_label   = NULL;
	  const enkf_config_node_type * config_node = enkf_config_get_node_ref(config , well_name);
	  
	  
	  if (strcmp(kw , "WELL") == 0) {
	    ASSERT_TOKENS("WELL" , active_tokens , 2);
	    enkf_obs_add_well_obs(enkf_obs , config_node , well_name , obs_label , config_file);
	  } else 
	    fprintf(stderr," ** Warning ** keyword:%s not recognized when parsing: %s - ignored \n",kw , config_file);
	  
	}
	util_free_string_list(token_list , tokens);
	free(line);
      }
    } while ( !at_eof );
    fclose(stream);
  }
  return enkf_obs;
}
