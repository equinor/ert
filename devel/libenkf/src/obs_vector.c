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


struct obs_vector_struct {
  obs_free_ftype       *freef;        /* Function used to free an observation node. */
  obs_get_ftype        *get_obs;      /* Function used to build the 'd' vector. */
  obs_meas_ftype       *measure;      /* Function used to measure on the state, and add to to the S matrix. */
  obs_activate_ftype   *activate;     /* This is used to activate / deactivate (parts of) the observation. */ 
  obs_type_check_ftype *type_check;   /* Used to to type_check the void pointers when installing a node. */
  obs_user_get_ftype   *user_get;     /* Function to get an observation based on KEY:INDEX input from user.*/
  
  obs_impl_type        obs_type; 
  char               * state_kw;
  void              ** nodes;       /* List of obs_node instances - NULL for all inactive report steps. */
  int                  size;        /* The number of report_steps. */
  int                  num_active;  /* The total number of timesteps where this observation is active (i.e. nodes[ ] != NULL) */
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
  int i;
  vector->nodes  = util_realloc(vector->nodes  , new_size * sizeof * vector->nodes  , __func__);
  for (i=vector->size; i < new_size; i++) 
    vector->nodes[i]  = NULL;

  vector->size  = new_size;
}



static obs_vector_type * obs_vector_alloc(obs_impl_type obs_type , const char * state_kw ,int num_reports) {
  obs_vector_type * vector = util_malloc(sizeof * vector , __func__);
  
  vector->freef      = NULL;
  vector->measure    = NULL;
  vector->get_obs    = NULL;
  vector->activate   = NULL;
  vector->type_check = NULL;
  vector->user_get   = NULL;
  
  switch (obs_type) {
  case(summary_obs):
    vector->freef      = summary_obs_free__;
    vector->measure    = summary_obs_measure__;
    vector->get_obs    = summary_obs_get_observations__;
    vector->type_check = summary_obs_is_instance__;
    vector->user_get   = summary_obs_user_get__;
    break;
  case(field_obs):
    vector->freef      = field_obs_free__;
    vector->measure    = field_obs_measure__;
    vector->get_obs    = field_obs_get_observations__;
    vector->type_check = field_obs_is_instance__;
    vector->user_get   = field_obs_user_get__;
    break;
  case(gen_obs):
    vector->freef      = gen_obs_free__;
    vector->measure    = gen_obs_measure__;
    vector->get_obs    = gen_obs_get_observations__;
    vector->type_check = gen_obs_is_instance__;
    vector->user_get   = gen_obs_user_get__;
    break;
  default:
    util_abort("%s: internal error - obs_type:%d not recognized \n",__func__ , obs_type);
  }
  
  vector->obs_type           = obs_type; 
  vector->state_kw           = util_alloc_string_copy( state_kw );
  vector->size               = 0;
  vector->nodes              = NULL;
  vector->num_active         = 0;
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
  return obs_vector->state_kw; 
}



void obs_vector_free(obs_vector_type * obs_vector) {
  int i;
  for (i=0; i < obs_vector->size; i++)
    if (obs_vector->nodes[i] != NULL) 
      obs_vector->freef(obs_vector->nodes[i]);
  
  util_safe_free(obs_vector->nodes);
  free(obs_vector->state_kw);
  free(obs_vector);
}


static void obs_vector_install_node(obs_vector_type * obs_vector , int index , void * node) {
  if (obs_vector->nodes[index] != NULL)
    util_abort("%s: node is already installed for index:%d. \n",__func__ , index);

  if (obs_vector->type_check(node)) {
    obs_vector->nodes[index] = node;
    obs_vector->num_active++;
  } else
    util_abort("%s: tried to insert obs_node of wrong type - aborting \n",__func__);
}





void obs_vector_delete_node(obs_vector_type * obs_vector , int index) {
  if (obs_vector->nodes[index] != NULL)
    util_abort("%s: missing node for index:%d. \n",__func__ , index);
  obs_vector->freef(obs_vector->nodes[index]);
  obs_vector->nodes[index] = NULL;
  obs_vector->num_active--;
}





/*****************************************************************/

int obs_vector_get_num_active(const obs_vector_type * vector) {
  return vector->num_active;
}


/**
   IFF - only one - report step is active this function will return
   that report step. If more than report step is active, the function
   ambigous, and will abort. Check with get_num_active first!
*/

int obs_vector_get_active_report_step(const obs_vector_type * vector) {
  if (vector->num_active == 1) {
    int active_step = -1;
    int i;
    for (i=0; i < vector->size; i++) {
      if (vector->nodes[i] != NULL) {
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
  if (index < 0 || index >= vector->size)
    util_abort("%s: index:%d invald. Limits: [0,%d) \n",__func__ , index , vector->size);

  if (vector->nodes[index] != NULL)
    return true;
  else
    return false;
}


/* 
   Will happily return NULL if index is not active. 
*/
void * obs_vector_iget_node(const obs_vector_type * vector, int index) {
  if (index < 0 || index >= vector->size)
    util_abort("%s: index:%d invald. Limits: [0,%d) \n",__func__ , index , vector->size);
  
  return vector->nodes[index]; 
}


void obs_vector_user_get(const obs_vector_type * obs_vector , const char * index_key , int report_step , double * value , double * std , bool * valid) {
  obs_vector->user_get(obs_vector->nodes[report_step] , index_key , value , std , valid);
}

/*
  This function returns the next active report step, starting with
  'prev_step + 1'. If no more active steps are found, it will return
  -1.
*/


int obs_vector_get_next_active_step(const obs_vector_type * obs_vector , int prev_step) {
  if (prev_step >= (obs_vector->size - 1))
    return -1;
  else {
    int next_step = prev_step + 1;
    while (obs_vector->nodes[next_step] == NULL && next_step < obs_vector->size)
      next_step++;

    if (next_step == obs_vector->size)
      return -1; /* No more active steps. */
    else
      return next_step;
  }
}


/*****************************************************************/


obs_vector_type * obs_vector_alloc_from_SUMMARY_OBSERVATION(const conf_instance_type * conf_instance , const history_type * history) {
  if(!conf_instance_is_of_class(conf_instance, "SUMMARY_OBSERVATION"))
    util_abort("%s: internal error. expected \"SUMMARY_OBSERVATION\" instance, got \"%s\".\n",
               __func__, conf_instance_get_class_name_ref(conf_instance) );
  
  {
    double       obs_value       = conf_instance_get_item_value_double(conf_instance, "VALUE" );
    double       obs_error       = conf_instance_get_item_value_double(conf_instance, "ERROR" );
    const char * sum_key         = conf_instance_get_item_value_ref(   conf_instance, "KEY"   );
    const char * obs_key         = conf_instance_get_name_ref(conf_instance);
    int          size            = history_get_num_restarts(          history          );
    obs_vector_type * obs_vector = obs_vector_alloc( summary_obs , sum_key , size );
    int          obs_restart_nr  = __conf_instance_get_restart_nr(conf_instance , obs_key , history , size);
    summary_obs_type * sum_obs   = summary_obs_alloc(sum_key , obs_value , obs_error);
    
    obs_vector_install_node( obs_vector , obs_restart_nr , sum_obs );
    return obs_vector;
  }
}




obs_vector_type * obs_vector_alloc_from_GENERAL_OBSERVATION(const conf_instance_type * conf_instance , const history_type * history, const ensemble_config_type * ensemble_config) {
  if(!conf_instance_is_of_class(conf_instance, "GENERAL_OBSERVATION"))
    util_abort("%s: internal error. expected \"GENERAL_OBSERVATION\" instance, got \"%s\".\n",
               __func__, conf_instance_get_class_name_ref(conf_instance) );
  
  {
    const char * obs_key         = conf_instance_get_name_ref(conf_instance);
    const char * obs_file        = conf_instance_get_item_value_ref(   conf_instance, "OBS_FILE" );              
    const char * state_kw        = conf_instance_get_item_value_ref(   conf_instance, "DATA" );              
    int          size            = history_get_num_restarts( history );
    obs_vector_type * obs_vector = obs_vector_alloc( gen_obs , state_kw , size );
    int          obs_restart_nr  = __conf_instance_get_restart_nr(conf_instance , obs_key , history , size);
    const char * index_file      = NULL;
    const char * index_list      = NULL;
    gen_obs_type * gen_obs;

    if (conf_instance_has_item(conf_instance , "INDEX_FILE"))
      index_file = conf_instance_get_item_value_ref(   conf_instance, "INDEX_FILE" );              

    if (conf_instance_has_item(conf_instance , "INDEX_LIST"))
      index_list = conf_instance_get_item_value_ref(   conf_instance, "INDEX_LIST" );              

    {
      const enkf_config_node_type * config_node  = ensemble_config_get_node( ensemble_config , state_kw);
      if (enkf_config_node_get_impl_type(config_node) == GEN_DATA) {
	gen_obs = gen_obs_alloc(obs_file , index_file , index_list);	
	obs_vector_install_node( obs_vector , obs_restart_nr , gen_obs );
      } else {
	enkf_impl_type impl_type = enkf_config_node_get_impl_type(config_node);
	util_abort("%s: %s has implementation type:\'%s\' - expected:\'%s\'.\n",__func__ , state_kw , enkf_types_get_impl_name(impl_type) , enkf_types_get_impl_name(GEN_DATA));
      }
    }
    
    return obs_vector;
  }
}





obs_vector_type * obs_vector_alloc_from_HISTORY_OBSERVATION(const conf_instance_type * conf_instance , const history_type * history) {
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
    obs_vector_type * obs_vector;  
    
    // Get time series data from history object and allocate
    history_alloc_time_series_from_summary_key(history, sum_key, &size, &value, &default_used);
    std = util_malloc(size * sizeof * std, __func__);
    obs_vector = obs_vector_alloc( summary_obs , sum_key , size );
    
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
    

    for (restart_nr = 0; restart_nr < size; restart_nr++) {
      if (!default_used[restart_nr]) {
	summary_obs_type * sum_obs  = summary_obs_alloc( sum_key , value[restart_nr] , std[restart_nr]);
	obs_vector_install_node( obs_vector , restart_nr , sum_obs );
      } 
    }
    
    free(std);
    free(value);
    free(default_used);
    
    return obs_vector;
  }
}




obs_vector_type * obs_vector_alloc_from_BLOCK_OBSERVATION(const conf_instance_type * conf_instance,const history_type * history, const ensemble_config_type * ensemble_config) {
  if(!conf_instance_is_of_class(conf_instance, "BLOCK_OBSERVATION"))
    util_abort("%s: internal error. expected \"BLOCK_OBSERVATION\" instance, got \"%s\".\n",
               __func__, conf_instance_get_class_name_ref(conf_instance) );

  {
    obs_vector_type * obs_vector;
    field_obs_type  * block_obs;

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
      block_obs  = field_obs_alloc(obs_label, field_config , field_name, num_obs_pts, obs_i, obs_j, obs_k, obs_value, obs_std);
    }
    obs_vector = obs_vector_alloc( field_obs , field_name , size );
    obs_vector_install_node( obs_vector , obs_restart_nr , block_obs);
    
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

void obs_vector_iget_observations(const obs_vector_type * obs_vector , int report_step , obs_data_type * obs_data) {
  if (obs_vector->nodes[report_step] != NULL)
    obs_vector->get_obs(obs_vector->nodes[report_step] , report_step , obs_data);
}


void obs_vector_measure(const obs_vector_type * obs_vector , int report_step ,const enkf_node_type * enkf_node ,  meas_vector_type * meas_vector) {
  if (obs_vector->nodes[report_step] != NULL)
    obs_vector->measure(obs_vector->nodes[report_step] , enkf_node_value_ptr(enkf_node) , meas_vector);
}




/*****************************************************************/

VOID_FREE(obs_vector)
