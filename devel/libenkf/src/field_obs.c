#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <stringlist.h>
#include <enkf_util.h>
#include <field_obs.h> 
#include <obs_data.h>
#include <meas_op.h>
#include <meas_vector.h>
#include <field_config.h>
#include <field.h>

#define FIELD_OBS_TYPE_ID 661098

struct field_obs_struct {
  int      __type_id;
  char   * field_name;   /** The state keyword for the observed field - PRESSURE / SWAT / PORO /...   */
  char   * obs_label;    /** A user provided label for the observation.      */
  int      size;         /** The number of field cells observed.             */
  int    * i;            /** The i indices of the observed cells.            */
  int    * j;            /** The j indices of the observed cells.            */
  int    * k;            /** The k indices of the observed cells.            */
  double * obs_value;    /** The observed values.                            */
  double * obs_std;      /** The standard deviation of the observations.     */
};


SAFE_CAST(field_obs , FIELD_OBS_TYPE_ID);
IS_INSTANCE(field_obs , FIELD_OBS_TYPE_ID);


/**
   The input vectors i,j,k should contain offset zero values.
*/
field_obs_type * field_obs_alloc(
  const char   * obs_label,
  const char   * field_name,
  int            size,
  const int    * i,
  const int    * j,
  const int    * k,
  const double * obs_value,
  const double * obs_std)
{
  
  field_obs_type * field_obs = util_malloc(sizeof * field_obs, __func__);
  field_obs->__type_id       = FIELD_OBS_TYPE_ID;
  field_obs->size            = size;
  field_obs->field_name      = util_alloc_string_copy(field_name);
  field_obs->obs_label       = util_alloc_string_copy(obs_label);
  field_obs->i               = util_alloc_copy(i         , size * sizeof * field_obs->i         , __func__);
  field_obs->j               = util_alloc_copy(j         , size * sizeof * field_obs->j         , __func__);
  field_obs->k               = util_alloc_copy(k         , size * sizeof * field_obs->k         , __func__);
  field_obs->obs_value       = util_alloc_copy(obs_value , size * sizeof * field_obs->obs_value , __func__);
  field_obs->obs_std         = util_alloc_copy(obs_std   , size * sizeof * field_obs->obs_value , __func__);
  
  return field_obs;
}



void field_obs_free(
  field_obs_type * field_obs)
{
  free(field_obs->i);
  free(field_obs->j);
  free(field_obs->k);
  free(field_obs->obs_value);
  free(field_obs->obs_std);
  free(field_obs->field_name);
  free(field_obs->obs_label);
  free(field_obs);
}






const char * field_obs_get_field_name(
  const field_obs_type * field_obs)
{
  return field_obs->field_name;
}




void field_obs_get_observations(
  const field_obs_type * field_obs,
  int                    restart_nr,
  obs_data_type        * obs_data)
{
  for (int i=0; i < field_obs->size; i++)
    obs_data_add(obs_data , field_obs->obs_value[i] , field_obs->obs_std[i] , field_obs->field_name);
}




void field_obs_measure(
  const field_obs_type * field_obs,
  const field_type     * field_state,
  meas_vector_type     * meas_vector)
{
  for (int i=0; i < field_obs->size; i++) {
    double value = field_ijk_get_double(field_state , field_obs->i[i] , field_obs->j[i] , field_obs->k[i]);
    meas_vector_add(meas_vector , value);
  }
}





//field_obs_type * field_obs_alloc_from_BLOCK_OBSERVATION(
//  const conf_instance_type * conf_instance,
//  const history_type       * history)
//{
//  if(!conf_instance_is_of_class(conf_instance, "BLOCK_OBSERVATION"))
//    util_abort("%s: internal error. expected \"BLOCK_OBSERVATION\" instance, got \"%s\".\n",
//               __func__, conf_instance_get_class_name_ref(conf_instance) );
//
//  field_obs_type * field_obs;
//  
//  const char * obs_label      = conf_instance_get_name_ref(conf_instance);
//  const char * field_name     = conf_instance_get_item_value_ref(conf_instance, "FIELD");
//
//  stringlist_type * obs_pt_keys = conf_instance_alloc_list_of_sub_instances_of_class_by_name(conf_instance, "OBS");
//  int               num_obs_pts = stringlist_get_size(obs_pt_keys);
//
//  double * obs_value = util_malloc(num_obs_pts * sizeof * obs_value, __func__);
//  double * obs_std   = util_malloc(num_obs_pts * sizeof * obs_std  , __func__);
//  int    * obs_i     = util_malloc(num_obs_pts * sizeof * obs_i    , __func__);
//  int    * obs_j     = util_malloc(num_obs_pts * sizeof * obs_j    , __func__);
//  int    * obs_k     = util_malloc(num_obs_pts * sizeof * obs_k    , __func__);
//
//
//  /** Get the time of the observation. Can be given as date, days or restart. */
//  if(conf_instance_has_item(conf_instance, "RESTART"))
//  {
//    int size = history_get_num_restarts(history);
//    obs_restart_nr = conf_instance_get_item_value_int(conf_instance, "RESTART");
//    if(obs_restart_nr > size)
//      util_abort("%s: Observation %s occurs at restart %i, but history file has only %i restarts.\n",
//                 __func__, obs_label, obs_restart_nr, size);
//  }
//  else if(conf_instance_has_item(conf_instance, "DATE"))
//  {
//    time_t obs_date = conf_instance_get_item_value_time_t(conf_instance, "DATE"  );
//    obs_restart_nr  = history_get_restart_nr_from_time_t(history, obs_date);
//  }
//  else if(conf_instance_has_item(conf_instance, "DAYS"))
//  {
//    double days = conf_instance_get_item_value_double(conf_instance, "DAYS");
//    obs_restart_nr = history_get_restart_nr_from_days(history, days);
//  }
//  else
//    util_abort("%s: Internal error. Invalid conf_instance?\n", __func__);
//
//  /** Build the observation. */
//  for(int obs_pt_nr = 0; obs_pt_nr < num_obs_pts; obs_pt_nr++)
//  {
//    const char * obs_key = stringlist_iget(obs_pt_keys, obs_pt_nr);
//    const conf_instance_type * obs_instance = conf_instance_get_sub_instance_ref(conf_instance, obs_key);
//
//    obs_value[obs_pt_nr] = conf_instance_get_item_value_double(obs_instance, "VALUE");
//    obs_std  [obs_pt_nr] = conf_instance_get_item_value_double(obs_instance, "ERROR");
//    obs_i    [obs_pt_nr] = conf_instance_get_item_value_int(   obs_instance, "I");
//    obs_j    [obs_pt_nr] = conf_instance_get_item_value_int(   obs_instance, "J");
//    obs_k    [obs_pt_nr] = conf_instance_get_item_value_int(   obs_instance, "K");
//  }
//
//  field_obs = field_obs_alloc(obs_label, field_name, obs_restart_nr, num_obs_pts, obs_i, obs_j, obs_k, obs_value, obs_std);
//
//  free(obs_value);
//  free(obs_std);
//  free(obs_i);
//  free(obs_j);
//  free(obs_k);
//
//  stringlist_free(obs_pt_keys);
//  
//  return field_obs;
//}


bool field_obs_fwrite(
  const field_obs_type * field_obs,
  FILE                 * stream) {
  util_fwrite_string(field_obs->field_name, stream);
  util_fwrite_string(field_obs->obs_label , stream);

  util_fwrite_int   (field_obs->size      , stream);

  util_fwrite       (field_obs->i         , sizeof *field_obs->i        , field_obs->size, stream, __func__);
  util_fwrite       (field_obs->j         , sizeof *field_obs->j        , field_obs->size, stream, __func__);
  util_fwrite       (field_obs->k         , sizeof *field_obs->k        , field_obs->size, stream, __func__);

  util_fwrite       (field_obs->obs_value , sizeof *field_obs->obs_value, field_obs->size, stream, __func__);
  util_fwrite       (field_obs->obs_std   , sizeof *field_obs->obs_value, field_obs->size, stream, __func__);

  return true;
}



void field_obs_fread(
  field_obs_type * field_obs,
  FILE           * stream)
{
  field_obs->field_name = util_fread_alloc_string(stream);
  field_obs->obs_label  = util_fread_alloc_string(stream);

  field_obs->size       = util_fread_int(         stream);

  util_fread       (field_obs->i         , sizeof *field_obs->i        , field_obs->size, stream, __func__);
  util_fread       (field_obs->j         , sizeof *field_obs->j        , field_obs->size, stream, __func__);
  util_fread       (field_obs->k         , sizeof *field_obs->k        , field_obs->size, stream, __func__);

  util_fread       (field_obs->obs_value , sizeof *field_obs->obs_value, field_obs->size, stream, __func__);
  util_fread       (field_obs->obs_std   , sizeof *field_obs->obs_value, field_obs->size, stream, __func__);
}



double field_obs_user_get(const field_obs_type * field_obs , const char * index_key , bool * valid) {
  int      length;
  int    * indices = util_sscanf_alloc_active_list(index_key, &length);

  if(length != 3)
    *valid = false;
  else
  {
    *valid = true;
    
    int i = indices[0] - 1;
    int j = indices[1] - 1;
    int k = indices[2] - 1;
  }

  free(indices);
  return 1.0;
}


/*****************************************************************/

VOID_FREAD(field_obs)
VOID_FWRITE(field_obs)
VOID_FREE(field_obs)
VOID_GET_OBS(field_obs)
VOID_MEASURE(field_obs , field)
VOID_USER_GET(field_obs)
