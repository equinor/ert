/**
   See the file README.obs for ducumentation of the various datatypes
   involved with observations/measurement/+++.
*/

#include <stdlib.h>
#include <stdio.h>
#include <util.h>
#include <enkf_macros.h>
#include <obs_node.h>
#include <enkf_types.h>
#include <time.h>
#include <stdbool.h>
#include <sched_file.h>
#include <enkf_util.h>
#include <enkf_node.h>
#include <summary_obs.h>
#include <field_obs.h>
#include <gen_obs.h>

typedef enum {obs_undef = 0, obs_active = 1 , obs_inactive = 2} obs_active_type;


struct obs_node_struct {
  const void         *obs;
  obs_free_ftype     *freef;
  obs_get_ftype      *get_obs;   /* Function used to build the 'd' vector. */
  obs_meas_ftype     *measure;   /* Function used to measure on the state, and add to to the S matrix. */
  obs_activate_ftype *activate;  /* This is used to activate / deactivate (parts of) the observation. */ 
  
  char               *state_kw;  /* This is used to look up the corresponding enkf_state object. */
  char               *obs_label; /* WTF? */
  int                 size;
  obs_active_type    *active;    
  bool                default_active;
  obs_impl_type       obs_type;
};



static void obs_node_set_active_mode(obs_node_type * obs_node , int first_report , int last_report , bool active) {
  int report_nr;
  obs_active_type active_mode;
  first_report   = util_int_max(0 , first_report);
  last_report    = util_int_min(last_report , obs_node->size - 1);
  if (last_report < first_report) {
    fprintf(stderr,"%s: last_report:%d is before first_report:%d - aborting \n",__func__ , last_report , first_report);
    abort();
  }

  if (active)
    active_mode = obs_active;
  else
    active_mode = obs_inactive;
  
  for (report_nr = first_report; report_nr <= last_report; report_nr++)
    obs_node->active[report_nr] = active_mode;
}



static void obs_node_set_active_mode_time_t(obs_node_type * obs_node , const sched_file_type * sched , time_t time1 , time_t time2 , bool active) {
  int report1, report2;

  report1 = sched_file_get_restart_nr_from_time_t(sched , time1); 
  report2 = sched_file_get_restart_nr_from_time_t(sched , time2); 
  
  obs_node_set_active_mode(obs_node , report1 , report2 , active);
}




static void obs_node_resize(obs_node_type * node , int new_size) {
  obs_active_type active_mode;
  int i;
  node->active = util_realloc(node->active , new_size * sizeof * node->active , __func__);
  if (node->default_active)
    active_mode = obs_active;
  else
    active_mode = obs_inactive;

  for (i=node->size; i < new_size; i++)
    node->active[i] = active_mode;

  node->size               = new_size;
}


/*****************************************************************/

obs_impl_type obs_node_get_impl_type(const obs_node_type * node) {
  return node->obs_type;
}



obs_node_type * obs_node_alloc(const void      	  * obs,
			       const char      	  * state_kw,
			       const char      	  * obs_label,
			       obs_impl_type        obs_type ,  
			       int             	    num_reports,
			       bool            	    default_active) {

  obs_node_type * node     = util_malloc( sizeof *node , __func__);
  node->obs                = obs;
  /**
     Starting by setting all function pointers to NULL.
  */
  node->freef    = NULL;
  node->measure  = NULL;
  node->get_obs  = NULL;
  node->activate = NULL;
  
  switch (obs_type) {
  case(summary_obs):
    node->freef   = summary_obs_free__;
    node->measure = summary_obs_measure__;
    node->get_obs = summary_obs_get_observations__;
    break;
  case(field_obs):
    node->freef   = field_obs_free__;
    node->measure = field_obs_measure__;
    node->get_obs = field_obs_get_observations__;
    break;
  case(gen_obs):
    node->freef   = gen_obs_free__;
    node->measure = gen_obs_measure__;
    node->get_obs = gen_obs_get_observations__;
    break;
  default:
    util_abort("%s: internal error - obs_type:%d not recognized \n",__func__ , obs_type);
  }
  
  
  node->size               = 0;
  node->active             = NULL;
  node->default_active     = default_active;
  node->obs_label          = util_alloc_string_copy(obs_label);
  node->state_kw           = util_alloc_string_copy(state_kw);
  node->obs_type           = obs_type;   
  obs_node_resize(node , num_reports); /* +1 here ?? Ohh  - these fucking +/- problems. */
  
  return node;
}



void obs_node_free(obs_node_type * node) {
  if (node->freef != NULL) node->freef( (void *) node->obs);
  if (node->obs_label != NULL) free(node->obs_label);
  free(node->state_kw);
  free(node->active);
  free(node);
}


void obs_node_get_observations(obs_node_type * node , int report_step, obs_data_type * obs_data) {
  if (node->get_obs != NULL)
    if (node->active[report_step] == obs_active) 
      node->get_obs(node->obs , report_step , obs_data);
}


void obs_node_measure(const obs_node_type * node , int report_step , const void * enkf_node , meas_vector_type * meas_vector) {
  if (node->active[report_step] == obs_active) 
    node->measure(node->obs , enkf_node_value_ptr(enkf_node) , meas_vector);
}


const void *  obs_node_get_ref(const obs_node_type * node) { 
  return node->obs; 
}

const char * obs_node_get_state_kw(const obs_node_type * node) { 
  return node->state_kw;
}



void obs_node_activate_report_step(obs_node_type * obs_node , int first_report , int last_report) {
  obs_node_set_active_mode(obs_node , first_report , last_report , true);
}


void obs_node_deactivate_report_step(obs_node_type * obs_node , int first_report , int last_report) {
  obs_node_set_active_mode(obs_node , first_report , last_report , false);
}


void obs_node_activate_time_t(obs_node_type * obs_node , const sched_file_type * sched_file , time_t time1 , time_t time2) {
  obs_node_set_active_mode_time_t(obs_node , sched_file , time1 , time2 , true);
}


void obs_node_deactivate_time_t(obs_node_type * obs_node , const sched_file_type * sched_file , time_t time1 , time_t time2) {
  obs_node_set_active_mode_time_t(obs_node , sched_file, time1 , time2, false);
}


VOID_FREE(obs_node)
