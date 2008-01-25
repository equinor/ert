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


typedef enum {obs_undef = 0, obs_active = 1 , obs_inactive = 2} obs_active_type;


struct obs_node_struct {
  const void       *obs;
  obs_free_ftype   *freef;
  obs_get_ftype    *get_obs;
  obs_meas_ftype   *measure;

  char             *obs_label;
  int               size;
  obs_active_type  *active;    
  bool              default_active;
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
  int status1 , status2;
  int report1 , report2;

  report1 = sched_file_time_t_to_report_step(sched , time1 , &status1); 
  report2 = sched_file_time_t_to_report_step(sched , time2 , &status2); 

  if (status1 == -1)
    report1 = 0;
  else if (status1 == 1) {
    int year , mday , mon;
    util_set_date_values(time1 , &mday , &mon , &year);
    fprintf(stderr,"%s: *Warning* time start time: %02d/%02d/%4d is after simulation end - aborting \n",__func__ , mday , mon , year);
    abort();
  }

  if (status1 == 1)
    report1 = obs_node->size - 1;
  else if (status1 == -1) {
    int year , mday , mon;
    util_set_date_values(time1 , &mday , &mon , &year);
    fprintf(stderr,"%s: *Warning* end time: %02d/%02d/%4d is before simulation start - aborting \n",__func__ , mday , mon , year);
    abort();
  }
  
  obs_node_set_active_mode(obs_node , report1 , report2 , active);
}




static void obs_node_resize(obs_node_type * node , int new_size) {
  obs_active_type active_mode;
  int i;
  node->active             = enkf_util_realloc(node->active , new_size * sizeof * node->active , __func__);
  if (node->default_active)
    active_mode = obs_active;
  else
    active_mode = obs_inactive;

  for (i=node->size; i < new_size; i++)
    node->active[i] = active_mode;

  node->size               = new_size;
}


/*****************************************************************/





obs_node_type * obs_node_alloc(const void      * obs,
			       const char      * obs_label,
			       int               num_reports,
			       bool              default_active,
			       obs_get_ftype   * get_obs,
			       obs_meas_ftype  * measure,
			       obs_free_ftype  * freef) {

  obs_node_type * node = malloc( sizeof *node);
  node->obs                = obs;
  node->freef              = freef;
  node->measure            = measure;
  node->get_obs            = get_obs;
  node->size               = 0;
  node->active             = NULL;
  node->default_active     = default_active;
  node->obs_label          = util_alloc_string_copy(obs_label);
  obs_node_resize(node , num_reports);
  
  return node;
}



void obs_node_free(obs_node_type * node) {
  if (node->freef != NULL) node->freef( (void *) node->obs);
  if (node->obs_label != NULL) free(node->obs_label);
  free(node->active);
  free(node);
}


void obs_node_get_observations(obs_node_type * node , int report_step, obs_data_type * obs_data) {
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
