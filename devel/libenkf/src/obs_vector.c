/**
   See the overview documentation of the observation system in enkf_obs.c
*/
#include <obs_vector.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <history.h>
#include <conf.h>
#include <enkf_fs.h>
#include <sched_file.h>
#include <util.h>
#include <time.h>
#include <summary_obs.h>
#include <field_obs.h>
#include <gen_obs.h>
#include <ensemble_config.h>
#include <msg.h>
#include <active_list.h>
#include <ecl_sum.h>
#include <vector.h>

#define OBS_VECTOR_TYPE_ID 120086

struct obs_vector_struct {
  UTIL_TYPE_ID_DECLARATION;
  obs_free_ftype       *freef;        /* Function used to free an observation node. */
  obs_get_ftype        *get_obs;      /* Function used to build the 'd' vector. */
  obs_meas_ftype       *measure;      /* Function used to measure on the state, and add to to the S matrix. */
  obs_user_get_ftype   *user_get;     /* Function to get an observation based on KEY:INDEX input from user.*/
  obs_chi2_ftype       *chi2;         /* Function to evaluate chi-squared for an observation. */ 
  

  vector_type                    * nodes; 
  char                           * obs_key;     /* The key this observation vector has in the enkf_obs layer. */ 
  enkf_config_node_type          * config_node; /* The config_node of the node type we are observing - shared reference */
  obs_impl_type    	           obs_type; 
  int              	           num_active;  /* The total number of timesteps where this observation is active (i.e. nodes[ ] != NULL) */
};


/*****************************************************************/


static int __conf_instance_get_restart_nr(const conf_instance_type * conf_instance, const char * obs_key , const history_type * history, int size) {
  int obs_restart_nr = -1;  /* To shut up compiler warning. */
  
  if(conf_instance_has_item(conf_instance, "RESTART")) {
    obs_restart_nr = conf_instance_get_item_value_int(conf_instance, "RESTART");
    if(obs_restart_nr > size)
      util_abort("%s: Observation %s occurs at restart %i, but history file has only %i restarts.\n", __func__, obs_key, obs_restart_nr, size);
  } else if(conf_instance_has_item(conf_instance, "DATE")) {
    time_t obs_date = conf_instance_get_item_value_time_t(conf_instance, "DATE"  );
    obs_restart_nr  = history_get_restart_nr_from_time_t(history, obs_date);
  } else if (conf_instance_has_item(conf_instance, "DAYS")) {
    double days = conf_instance_get_item_value_double(conf_instance, "DAYS");
    obs_restart_nr = history_get_restart_nr_from_days(history, days);
  }  else
    util_abort("%s: Internal error. Invalid conf_instance?\n", __func__);
  
  return obs_restart_nr;
}



/*****************************************************************/


static void obs_vector_resize(obs_vector_type * vector , int new_size) {
  int current_size = vector_get_size( vector->nodes );
  int i;
  for (i=current_size; i < new_size; i++) 
    vector_append_ref( vector->nodes , NULL);
  
}



obs_vector_type * obs_vector_alloc(obs_impl_type obs_type , const char * obs_key , enkf_config_node_type * config_node , int num_reports) {
  obs_vector_type * vector = util_malloc(sizeof * vector , __func__);
  
  UTIL_TYPE_ID_INIT( vector , OBS_VECTOR_TYPE_ID);
  vector->freef      = NULL;
  vector->measure    = NULL;
  vector->get_obs    = NULL;
  vector->user_get   = NULL;
  vector->chi2       = NULL;
  
  switch (obs_type) {
  case(SUMMARY_OBS):
    vector->freef      = summary_obs_free__;
    vector->measure    = summary_obs_measure__;
    vector->get_obs    = summary_obs_get_observations__;
    vector->user_get   = summary_obs_user_get__;
    vector->chi2       = summary_obs_chi2__;
    break;
  case(FIELD_OBS):
    vector->freef      = field_obs_free__;
    vector->measure    = field_obs_measure__;
    vector->get_obs    = field_obs_get_observations__;
    vector->user_get   = field_obs_user_get__;
    vector->chi2       = field_obs_chi2__;
    break;
  case(GEN_OBS):
    vector->freef      = gen_obs_free__;
    vector->measure    = gen_obs_measure__;
    vector->get_obs    = gen_obs_get_observations__;
    vector->user_get   = gen_obs_user_get__;
    vector->chi2       = gen_obs_chi2__; 
    break;
  default:
    util_abort("%s: internal error - obs_type:%d not recognized \n",__func__ , obs_type);
  }
  
  vector->obs_type           = obs_type;
  vector->config_node        = config_node;
  vector->obs_key            = util_alloc_string_copy( obs_key );
  vector->num_active         = 0;
  vector->nodes              = vector_alloc_new();
  obs_vector_resize(vector , num_reports); /* +1 here ?? Ohh  - these fucking +/- problems. */

  return vector;
}

obs_impl_type obs_vector_get_impl_type(const obs_vector_type * obs_vector) {
  return obs_vector->obs_type;
}


/**
   This is the key for the enkf_node which this observation is
   'looking at'. I.e. if this observation is an RFT pressure
   measurement, this function will return "PRESSURE".
*/

const char * obs_vector_get_state_kw(const obs_vector_type * obs_vector) {
  return enkf_config_node_get_key( obs_vector->config_node );
}


enkf_config_node_type * obs_vector_get_config_node(obs_vector_type * obs_vector) {
  return obs_vector->config_node;
}



void obs_vector_free(obs_vector_type * obs_vector) {
  vector_free( obs_vector->nodes );
  free(obs_vector->obs_key);
  free(obs_vector);
}


static void obs_vector_assert_node_type( const obs_vector_type * obs_vector , const void * node ) {
  bool type_OK;
  switch (obs_vector->obs_type) {
  case(SUMMARY_OBS):
    type_OK = summary_obs_is_instance( node );
    break;
  case(FIELD_OBS):
    type_OK = field_obs_is_instance( node );
    break;
  case(GEN_OBS):
    type_OK = gen_obs_is_instance( node );
    break;
  default:
    util_abort("%s: What the fuck? \n",__func__);
    type_OK = false;
  }
  if (!type_OK) 
    util_abort("%s: Type mismatch when trying to add observation node to observation vector \n",__func__);
}




void obs_vector_del_node(obs_vector_type * obs_vector , int index) {
  if (vector_iget_const( obs_vector->nodes , index ) != NULL) {
    vector_iset_ref( obs_vector->nodes , index , NULL);  /* Clear current content. */
    obs_vector->num_active--;
  }
}

/**
   This function will clear (and free) all the summary_obs / gen_obs /
   field_obs instances which have been installed in the vector;
   however the vector itself is retained with keys, function pointers
   and so on.
*/

void obs_vector_clear_nodes( obs_vector_type * obs_vector ) {
  vector_clear( obs_vector->nodes );
  obs_vector->num_active = 0;
}



static void obs_vector_install_node(obs_vector_type * obs_vector , int index , void * node) {
  obs_vector_assert_node_type( obs_vector , node );
  {
    if (vector_iget_const( obs_vector->nodes , index ) == NULL)
      obs_vector->num_active++;
    
    vector_iset_owned_ref( obs_vector->nodes , index , node , obs_vector->freef );
  }
}



/**
   Observe that @sumamry_key is the key used to look up the
   corresponding simulated value in the ensemble, and not the
   observation key - the two can be different.
*/

void obs_vector_add_summary_obs( obs_vector_type * obs_vector , int obs_index , const char * summary_key , double value , double std) {
  summary_obs_type * summary_obs = summary_obs_alloc( summary_key , value , std );
  obs_vector_install_node( obs_vector , obs_index , summary_obs );
}








/*****************************************************************/

int obs_vector_get_num_active(const obs_vector_type * vector) {
  return vector->num_active;
}


/**
   IFF - only one - report step is active this function will return
   that report step. If more than report step is active, the function
   is ambigous, and will fail HARD. Check with get_num_active first!
*/

int obs_vector_get_active_report_step(const obs_vector_type * vector) {
  if (vector->num_active == 1) {
    int active_step = -1;
    int i;
    for (i=0; i < vector_get_size(vector->nodes); i++) {
      void * obs_node = vector_iget( vector->nodes , i);
      if (obs_node != NULL) {
	if (active_step >= 0)
	  util_abort("%s: internal error - mismatch in obs_vector->nodes and obs_vector->num_active \n",__func__);
	active_step = i;
      }
    }
    if (active_step < 0)
      util_abort("%s: internal error - mismatch in obs_vector->nodes and obs_vector->num_active \n",__func__);
    
    return active_step;
  } else {
    util_abort("%s: when calling this function the number of active report steps MUST BE 1 - you had: %d \n",__func__ , vector->num_active);
    return 0; /* Comiler shut up. */
  }
}



bool obs_vector_iget_active(const obs_vector_type * vector, int index) {
  /* We accept this ... */
  if (index >= vector_get_size( vector->nodes ))
    return false;
  
  {
    void * obs_data = vector_iget( vector->nodes , index );
    if (obs_data != NULL) 
      return true;
    else
      return false;
  }
}


/* 
   Will happily return NULL if index is not active. 
*/
void * obs_vector_iget_node(const obs_vector_type * vector, int index) {
  return vector_iget( vector->nodes , index );
}


void obs_vector_user_get(const obs_vector_type * obs_vector , const char * index_key , int report_step , double * value , double * std , bool * valid) {
  obs_vector->user_get(obs_vector_iget_node(obs_vector , report_step) , index_key , value , std , valid);
}

/*
  This function returns the next active report step, starting with
  'prev_step + 1'. If no more active steps are found, it will return
  -1.
*/

int obs_vector_get_next_active_step(const obs_vector_type * obs_vector , int prev_step) {
  if (prev_step >= (vector_get_size(obs_vector->nodes) - 1))
    return -1;
  else {
    int size = vector_get_size( obs_vector->nodes );
    int next_step = prev_step + 1;
    while (( next_step < size) && (obs_vector_iget_node(obs_vector , next_step) == NULL))
      next_step++;

    if (next_step == size)
      return -1; /* No more active steps. */
    else
      return next_step;
  }
}


/*****************************************************************/


void obs_vector_load_from_SUMMARY_OBSERVATION(obs_vector_type * obs_vector , const conf_instance_type * conf_instance , const history_type * history, ensemble_config_type * ensemble_config) {
  if(!conf_instance_is_of_class(conf_instance, "SUMMARY_OBSERVATION"))
    util_abort("%s: internal error. expected \"SUMMARY_OBSERVATION\" instance, got \"%s\".\n",
               __func__, conf_instance_get_class_name_ref(conf_instance) );
  
  {
    double       obs_value       = conf_instance_get_item_value_double(conf_instance, "VALUE" );
    double       obs_error       = conf_instance_get_item_value_double(conf_instance, "ERROR" );
    const char * sum_key         = conf_instance_get_item_value_ref(   conf_instance, "KEY"   );
    const char * obs_key         = conf_instance_get_name_ref(conf_instance);
    int          size            = history_get_num_restarts(          history          );
    int          obs_restart_nr  = __conf_instance_get_restart_nr(conf_instance , obs_key , history , size);
    summary_obs_type * sum_obs;

    obs_vector_add_summary_obs( obs_vector , obs_restart_nr , sum_key , obs_value , obs_error );
  }
}




obs_vector_type * obs_vector_alloc_from_GENERAL_OBSERVATION(const conf_instance_type * conf_instance , const history_type * history, const ensemble_config_type * ensemble_config) {
  if(!conf_instance_is_of_class(conf_instance, "GENERAL_OBSERVATION"))
    util_abort("%s: internal error. expected \"GENERAL_OBSERVATION\" instance, got \"%s\".\n",
               __func__, conf_instance_get_class_name_ref(conf_instance) );
  
  {
    const char * obs_key         = conf_instance_get_name_ref(conf_instance);
    const char * state_kw        = conf_instance_get_item_value_ref(   conf_instance, "DATA" );              
    int          size            = history_get_num_restarts( history );
    obs_vector_type * obs_vector = obs_vector_alloc( GEN_OBS , obs_key , ensemble_config_get_node(ensemble_config , state_kw ) , size );
    int          obs_restart_nr  = __conf_instance_get_restart_nr(conf_instance , obs_key , history , size);
    const char * index_file      = NULL;
    const char * index_list      = NULL;
    const char * obs_file        = NULL;
    
    if (conf_instance_has_item(conf_instance , "INDEX_FILE"))
      index_file = conf_instance_get_item_value_ref(   conf_instance, "INDEX_FILE" );              

    if (conf_instance_has_item(conf_instance , "INDEX_LIST"))
      index_list = conf_instance_get_item_value_ref(   conf_instance, "INDEX_LIST" );              

    if (conf_instance_has_item(conf_instance , "OBS_FILE"))
      obs_file = conf_instance_get_item_value_ref(   conf_instance, "OBS_FILE" );              
    
    
    {
      const enkf_config_node_type * config_node  = ensemble_config_get_node( ensemble_config , state_kw);
      if (enkf_config_node_get_impl_type(config_node) == GEN_DATA) {
	double scalar_error = -1;
	double scalar_value = -1;
	gen_obs_type * gen_obs ;

	if (conf_instance_has_item(conf_instance , "VALUE")) {
	  scalar_value = conf_instance_get_item_value_double(conf_instance , "VALUE");
	  scalar_error = conf_instance_get_item_value_double(conf_instance , "ERROR");
	}

	/** The config system has ensured that we have either OBS_FILE or (VALUE and ERROR). */
	gen_obs = gen_obs_alloc(obs_key , obs_file , scalar_value , scalar_error , index_file , index_list);	
	obs_vector_install_node( obs_vector , obs_restart_nr , gen_obs );
      } else {
	enkf_impl_type impl_type = enkf_config_node_get_impl_type(config_node);
	util_abort("%s: %s has implementation type:\'%s\' - expected:\'%s\'.\n",__func__ , state_kw , enkf_types_get_impl_name(impl_type) , enkf_types_get_impl_name(GEN_DATA));
      }
    }
    
    return obs_vector;
  }
}



// Should check the refcase for key - if it is != NULL.

 void obs_vector_load_from_HISTORY_OBSERVATION(obs_vector_type * obs_vector , const conf_instance_type * conf_instance , const history_type * history , ensemble_config_type * ensemble_config, double std_cutoff) {
  if(!conf_instance_is_of_class(conf_instance, "HISTORY_OBSERVATION"))
    util_abort("%s: internal error. expected \"HISTORY_OBSERVATION\" instance, got \"%s\".\n",__func__, conf_instance_get_class_name_ref(conf_instance) );
  
  {
    int          size , restart_nr;
    double     * value;
    double     * std;
    bool       * default_used;
    
    double         error      = conf_instance_get_item_value_double(conf_instance, "ERROR"     );
    double         error_min  = conf_instance_get_item_value_double(conf_instance, "ERROR_MIN" );
    const char *   error_mode = conf_instance_get_item_value_ref(   conf_instance, "ERROR_MODE");
    const char *   sum_key    = conf_instance_get_name_ref(         conf_instance              );
    
    // Get time series data from history object and allocate
    history_alloc_time_series_from_summary_key(history, sum_key, &size, &value, &default_used);
    
    
    std = util_malloc(size * sizeof * std, __func__);
    
    // Create  the standard deviation vector
    if(strcmp(error_mode, "ABS") == 0) {
      for( restart_nr = 0; restart_nr < size; restart_nr++)
       std[restart_nr] = error;
    } else if(strcmp(error_mode, "REL") == 0) {
      for( restart_nr = 0; restart_nr < size; restart_nr++)
       std[restart_nr] = error * abs(value[restart_nr]);
    } 
    else if(strcmp(error_mode, "RELMIN") == 0) {
      for(restart_nr = 0; restart_nr < size; restart_nr++) {
       std[restart_nr] = error * abs(value[restart_nr]);
       if(std[restart_nr] < error_min)
         std[restart_nr] = error_min;
      }
    } else
      util_abort("%s: Internal error. Unknown error mode \"%s\"\n", __func__, error_mode);
    
    // Handle SEGMENTs that can customize the observation error.
    stringlist_type * segment_keys = conf_instance_alloc_list_of_sub_instances_of_class_by_name(conf_instance, "SEGMENT");
    stringlist_sort( segment_keys , NULL );

    int num_segments = stringlist_get_size(segment_keys);

    for(int segment_nr = 0; segment_nr < num_segments; segment_nr++)
    {
      const char * segment_name = stringlist_iget(segment_keys, segment_nr);
      const conf_instance_type * segment_conf = conf_instance_get_sub_instance_ref(conf_instance, segment_name);

      int start                         = conf_instance_get_item_value_int(   segment_conf, "START"     );
      int stop                          = conf_instance_get_item_value_int(   segment_conf, "STOP"      );
      double         error_segment      = conf_instance_get_item_value_double(segment_conf, "ERROR"     );
      double         error_min_segment  = conf_instance_get_item_value_double(segment_conf, "ERROR_MIN" );
      const char *   error_mode_segment = conf_instance_get_item_value_ref(   segment_conf, "ERROR_MODE");

      if(start < 0)
      {
        printf("%s: WARNING - Segment out of bounds. Truncating start of segment to 0.\n", __func__);
        start = 0;
      }

      if(stop >= size)
      {
        printf("%s: WARNING - Segment out of bounds. Truncating end of segment to %d.\n", __func__, size - 1);
        stop = size -1;
      }

      if(start > stop)
      {
        printf("%s: WARNING - Segment start after stop. Truncating end of segment to %d.\n", __func__, start );
        stop = start;
      }

      // Create  the standard deviation vector
      if(strcmp(error_mode_segment, "ABS") == 0) {
        for( restart_nr = start; restart_nr <= stop; restart_nr++)
         std[restart_nr] = error_segment;
      } else if(strcmp(error_mode_segment, "REL") == 0) {
        for( restart_nr = start; restart_nr <= stop; restart_nr++)
         std[restart_nr] = error_segment * abs(value[restart_nr]);
      } 
      else if(strcmp(error_mode_segment, "RELMIN") == 0) {
        for(restart_nr = start; restart_nr <= stop ; restart_nr++) {
          std[restart_nr] = error_segment * abs(value[restart_nr]);
          if(std[restart_nr] < error_min_segment)
            std[restart_nr] = error_min_segment;
        }
      } else
        util_abort("%s: Internal error. Unknown error mode \"%s\"\n", __func__, error_mode);
    }

    stringlist_free(segment_keys);
    

    for (restart_nr = 0; restart_nr < size; restart_nr++) {
      if (!default_used[restart_nr]) {
        if (std[restart_nr] > std_cutoff) 
          obs_vector_add_summary_obs( obs_vector , restart_nr , sum_key , value[restart_nr] , std[restart_nr]);
        else 
          fprintf(stderr,"** Warning: to small observation error in observation %s:%d - ignored. \n", sum_key , restart_nr);
      } 
    }
    
    free(std);
    free(value);
    free(default_used);
  }
}




obs_vector_type * obs_vector_alloc_from_BLOCK_OBSERVATION(const conf_instance_type * conf_instance,const history_type * history, const ensemble_config_type * ensemble_config) {
  if(!conf_instance_is_of_class(conf_instance, "BLOCK_OBSERVATION"))
    util_abort("%s: internal error. expected \"BLOCK_OBSERVATION\" instance, got \"%s\".\n",
               __func__, conf_instance_get_class_name_ref(conf_instance) );

  {
    obs_vector_type * obs_vector;
    
    int          size            = history_get_num_restarts( history );
    int          obs_restart_nr ;
    const char * obs_label      = conf_instance_get_name_ref(conf_instance);
    const char * field_name     = conf_instance_get_item_value_ref(conf_instance, "FIELD");
    
    stringlist_type * obs_pt_keys = conf_instance_alloc_list_of_sub_instances_of_class_by_name(conf_instance, "OBS");
    int               num_obs_pts = stringlist_get_size(obs_pt_keys);
    
    double * obs_value = util_malloc(num_obs_pts * sizeof * obs_value, __func__);
    double * obs_std   = util_malloc(num_obs_pts * sizeof * obs_std  , __func__);
    int    * obs_i     = util_malloc(num_obs_pts * sizeof * obs_i    , __func__);
    int    * obs_j     = util_malloc(num_obs_pts * sizeof * obs_j    , __func__);
    int    * obs_k     = util_malloc(num_obs_pts * sizeof * obs_k    , __func__);

    obs_restart_nr = __conf_instance_get_restart_nr(conf_instance , obs_label , history , size);  
    
    /** Build the observation. */
    for(int obs_pt_nr = 0; obs_pt_nr < num_obs_pts; obs_pt_nr++) {
      const char * obs_key = stringlist_iget(obs_pt_keys, obs_pt_nr);
      const conf_instance_type * obs_instance = conf_instance_get_sub_instance_ref(conf_instance, obs_key);
      
      obs_value[obs_pt_nr] = conf_instance_get_item_value_double(obs_instance, "VALUE");
      obs_std  [obs_pt_nr] = conf_instance_get_item_value_double(obs_instance, "ERROR");
      
      /**
	 The input values i,j,k come from the user, and are offset 1. They
	 are immediately shifted with -1 to become C-based offset zero.
      */
      obs_i    [obs_pt_nr] = conf_instance_get_item_value_int(   obs_instance, "I") - 1;
      obs_j    [obs_pt_nr] = conf_instance_get_item_value_int(   obs_instance, "J") - 1;
      obs_k    [obs_pt_nr] = conf_instance_get_item_value_int(   obs_instance, "K") - 1;
    }
    
    {
      const enkf_config_node_type * config_node  = ensemble_config_get_node( ensemble_config , field_name);
      const field_config_type     * field_config = enkf_config_node_get_ref( config_node ); 
      field_obs_type * block_obs  = field_obs_alloc(obs_label, field_config , field_name, num_obs_pts, obs_i, obs_j, obs_k, obs_value, obs_std);
      obs_vector = obs_vector_alloc( FIELD_OBS , obs_label , ensemble_config_get_node(ensemble_config , field_name) , size );
      
      obs_vector_install_node( obs_vector , obs_restart_nr , block_obs);
    }
    
    free(obs_value);
    free(obs_std);
    free(obs_i);
    free(obs_j);
    free(obs_k);
    
    stringlist_free(obs_pt_keys);
  
    return obs_vector;
  }
}
/*****************************************************************/

void obs_vector_iget_observations(const obs_vector_type * obs_vector , int report_step , obs_data_type * obs_data, const active_list_type * active_list) {
  void * obs_node = vector_iget( obs_vector->nodes , report_step );
  if ( obs_node != NULL) 
    obs_vector->get_obs(obs_node , report_step , obs_data , active_list);
}


void obs_vector_measure(const obs_vector_type * obs_vector , int report_step ,const enkf_node_type * enkf_node ,  meas_vector_type * meas_vector, const active_list_type * active_list) {
  void * obs_node = vector_iget( obs_vector->nodes , report_step );
  if ( obs_node != NULL) 
    obs_vector->measure(obs_node , enkf_node_value_ptr(enkf_node) , meas_vector, active_list);
}



/*****************************************************************/
/** Here comes many different functions for misfit calculations. */

/**
   This is the lowest level function:

   * It is checked that the obs_vector is active for the actual report
     step; if it is not active 0.0 is returned without any further
     ado.

   * It is assumed the enkf_node_instance contains valid data for this
     report_step. This is not checked in this function, and is the
     responsability of the calling scope.

   * The underlying chi2 function will do a type-check of node - and
     fail hard if it is not correct.

*/


static double obs_vector_chi2__(const obs_vector_type * obs_vector , int report_step , const enkf_node_type * node) { 
  void * obs_node = vector_iget( obs_vector->nodes , report_step );
  if ( obs_node != NULL) 
    return obs_vector->chi2( obs_node , enkf_node_value_ptr( node ));
  else
    return 0.0;  /* Observation not active for this report step. */
}



/**
   This function will load the node from the filesystem. It will load
   the state (analyzed | forecast) indicated by load_state. If
   load_state == both, it will first try the analyzed and the
   subsequently the forecast. The return value will be state actually
   loaded, and 'undefined' if nothing was loaded.
*/
   
static state_enum obs_vector_load_node__(enkf_fs_type * fs , enkf_node_type * node, state_enum load_state , int report_step , int iens) {
  state_enum state = UNDEFINED;

  if (load_state == FORECAST) {
    if (enkf_fs_has_node(fs , enkf_node_get_config(node) , report_step , iens , FORECAST)) {
      enkf_fs_fread_node( fs , node , report_step , iens , FORECAST );
      state = FORECAST;    
    }
  } else if (load_state == ANALYZED) {
    if (enkf_fs_has_node(fs , enkf_node_get_config(node) , report_step , iens , ANALYZED)) {
      enkf_fs_fread_node( fs , node , report_step , iens , FORECAST );
      state = ANALYZED;    
    }
  } else if (load_state == BOTH) {
    /* Trying analyzed first */
    if (enkf_fs_has_node(fs , enkf_node_get_config(node) , report_step , iens , ANALYZED)) {
      enkf_fs_fread_node( fs , node , report_step , iens , ANALYZED);
      state = ANALYZED;
    } else if (enkf_fs_has_node(fs , enkf_node_get_config(node) , report_step , iens , FORECAST)) {
      enkf_fs_fread_node( fs , node , report_step , iens , FORECAST );
      state = FORECAST;
    }
  }

  if (state == UNDEFINED)
    fprintf(stderr , "** Warning could not locate: %s / %d / %d for misfit calculations - defaulting to ZERO misfit. \n",enkf_node_get_key(node) , report_step , iens);

  return state;
}



double obs_vector_chi2(const obs_vector_type * obs_vector , enkf_fs_type * fs , int report_step , int iens , state_enum load_state) {
  enkf_node_type * enkf_node = enkf_node_alloc( obs_vector->config_node );
  double chi2 = 0;

  if (obs_vector_load_node__(fs , enkf_node , load_state , report_step , iens) != UNDEFINED) 
    chi2 = obs_vector_chi2__(obs_vector , report_step , enkf_node);
  
  enkf_node_free( enkf_node );
  return chi2;
}




/**
   This function will evaluate the chi2 for the ensemble members
   [iens1,iens2) and report steps [step1,step2).

   Observe that the chi2 pointer is assumed to be allocated for the
   complete ensemble, altough this function only operates on part of
   it.
*/


void obs_vector_ensemble_chi2(const obs_vector_type * obs_vector , enkf_fs_type * fs, int step1 , int step2 , int iens1 , int iens2 , state_enum load_state , double ** chi2) {
  int step;

  enkf_node_type * enkf_node = enkf_node_alloc( obs_vector->config_node );
  for (step = step1; step < step2; step++) {
    int iens;
    if (vector_iget( obs_vector->nodes , step) != NULL) {
      for (iens = iens1; iens < iens2; iens++) {
	if (obs_vector_load_node__(fs , enkf_node , load_state ,step , iens) != UNDEFINED) 
	  chi2[step][iens] = obs_vector_chi2__(obs_vector , step , enkf_node);
	else
	  chi2[step][iens] = 0;
      }
    } else {
      for (iens = iens1; iens < iens2; iens++) 
	chi2[step][iens] = 0;
    }
  }
  enkf_node_free( enkf_node );
}


/**
   This function will evaluate the total chi2 for one ensemble member
   (i.e. sum over report steps).
*/


double obs_vector_total_chi2(const obs_vector_type * obs_vector , enkf_fs_type * fs , int iens, state_enum load_state) {
  int report_step;
  double sum_chi2 = 0;
  enkf_node_type * enkf_node = enkf_node_alloc( obs_vector->config_node );

  for (report_step = 0; report_step < vector_get_size( obs_vector->nodes ); report_step++) {
    if (vector_iget(obs_vector->nodes , report_step) != NULL) {
      if (obs_vector_load_node__(fs , enkf_node , load_state , report_step , iens) != UNDEFINED) 
	sum_chi2 += obs_vector_chi2__(obs_vector , report_step , enkf_node);
    }
  }
  enkf_node_free( enkf_node );
  return sum_chi2;
}


/** 
   This function will sum up all timesteps of the obs_vector, for all ensemble members.
*/

void obs_vector_ensemble_total_chi2(const obs_vector_type * obs_vector , enkf_fs_type * fs , int ens_size , state_enum load_state , double * sum_chi2) {
  const bool verbose = true;
  msg_type * msg;
  int report_step;
  int iens;
  char * msg_text = NULL;
  for (iens = 0; iens < ens_size; iens++)
    sum_chi2[iens] = 0;

  if (verbose) {
    msg = msg_alloc("Observation: ");
    msg_show(msg);
  }

  enkf_node_type * enkf_node = enkf_node_alloc( obs_vector->config_node );
  for (report_step = 0; report_step < vector_get_size( obs_vector->nodes); report_step++) { 
    if (verbose) {
      msg_text = util_realloc_sprintf( msg_text , "%s[%03d]" , obs_vector->obs_key , report_step);
      msg_update(msg , msg_text);
    }
    if (vector_iget(obs_vector->nodes , report_step) != NULL) {
      for (iens = 0; iens < ens_size; iens++) {
	if (obs_vector_load_node__(fs , enkf_node , load_state , report_step , iens) != UNDEFINED) 
	  sum_chi2[iens] += obs_vector_chi2__(obs_vector , report_step , enkf_node);
      }
    }
  }
  enkf_node_free( enkf_node );
  if (verbose) {
    msg_free(msg , true);
    util_safe_free( msg_text );
  }
}

const char * obs_vector_get_obs_key( const obs_vector_type * obs_vector) {
  return obs_vector->obs_key;
}


/*****************************************************************/



UTIL_SAFE_CAST_FUNCTION(obs_vector , OBS_VECTOR_TYPE_ID)
VOID_FREE(obs_vector)
     
