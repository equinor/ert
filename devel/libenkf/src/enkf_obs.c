/**
   See the file README.obs for ducumentation of the varios datatypes
   involved with observations/measurement/+++.
*/

#include <stdlib.h>
#include <hash.h>
#include <util.h>
#include <conf.h>
#include <enkf_obs.h>
#include <summary_obs.h>
#include <field_obs.h>




struct enkf_obs_struct {
  /** A hash of obs_node_types indexed by user provided keys. */
  hash_type              * obs_hash; 
};


/** TODO This static function header shall be removed when the configuration is unified.... */
static conf_class_type * enkf_obs_get_obs_conf_class();


enkf_obs_type * enkf_obs_alloc(
)
{
  enkf_obs_type * enkf_obs = util_malloc(sizeof * enkf_obs, __func__);
  enkf_obs->obs_hash       = hash_alloc();
  return enkf_obs;
}



void enkf_obs_free(
  enkf_obs_type * enkf_obs)
{
  hash_free(enkf_obs->obs_hash);
  free(enkf_obs);
}


  
void enkf_obs_add_obs(
  enkf_obs_type       * enkf_obs,
  const char          * key ,
  const obs_node_type * node)
{
  if (hash_has_key(enkf_obs->obs_hash , key))
    util_abort("%s: Observation with key:%s already added.\n",__func__ , key);
  hash_insert_hash_owned_ref(enkf_obs->obs_hash , key , node , obs_node_free__);
}



void enkf_obs_get_observations(
  enkf_obs_type * enkf_obs ,
  int             report_step,
  obs_data_type * obs_data)
{
  char ** obs_keys = hash_alloc_keylist(enkf_obs->obs_hash);
  int iobs;

  obs_data_reset(obs_data);
  for (iobs = 0; iobs < hash_get_size(enkf_obs->obs_hash); iobs++) {
    obs_node_type * obs_node = hash_get(enkf_obs->obs_hash , obs_keys[iobs]);
    obs_node_get_observations(obs_node , report_step , obs_data);
  }
  util_free_stringlist( obs_keys , hash_get_size(enkf_obs->obs_hash));
}



void enkf_obs_measure_on_ensemble(
        const enkf_obs_type    * enkf_obs,
        enkf_fs_type           * fs,
        int                      report_step,
        state_enum               state,
        int                      ens_size,
        const enkf_state_type ** ensemble ,
        meas_matrix_type       * meas_matrix)
{
  char **obs_keys = hash_alloc_keylist(enkf_obs->obs_hash);
  int iobs;
  for (iobs = 0; iobs < hash_get_size(enkf_obs->obs_hash); iobs++)
  {
    const char * kw = obs_keys[iobs];
    {
      obs_node_type  * obs_node  = hash_get(enkf_obs->obs_hash , kw);
      int iens;
      for (iens = 0; iens < ens_size; iens++) {
        enkf_node_type * enkf_node = enkf_state_get_node(ensemble[iens] , obs_node_get_state_kw(obs_node));
        meas_vector_type * meas_vector = meas_matrix_iget_vector(meas_matrix , iens);
        enkf_fs_fread_node(fs , enkf_node , report_step , iens , state);       /* Hardcoded to measure on the forecast */
        obs_node_measure(obs_node , report_step , enkf_node , meas_vector);
      }
    }
  }
  util_free_stringlist( obs_keys , hash_get_size( enkf_obs->obs_hash ));
}



/** TODO
    When configuration has been unified, this should take a conf_instance_type, not a config_file.
*/
enkf_obs_type * enkf_obs_fscanf_alloc(
  const char         * config_file,
  const history_type * hist) 
{
  int num_restarts = history_get_num_restarts(hist);

  enkf_obs_type      * enkf_obs        = enkf_obs_alloc();
  if(config_file == NULL)
    return enkf_obs;
  conf_class_type    * enkf_conf_class = enkf_obs_get_obs_conf_class();
  conf_instance_type * enkf_conf       = conf_instance_alloc_from_file(enkf_conf_class, "enkf_conf", config_file); 

  
  if(conf_instance_validate(enkf_conf) == false)
  {
    util_abort("Can not proceed with this configuration.\n");
  }
  



  /** Handle HISTORY_OBSERVATION instances. */
  {
    stringlist_type * hist_obs_keys = conf_instance_alloc_list_of_sub_instances_of_class_by_name(enkf_conf, "HISTORY_OBSERVATION");
    int               num_hist_obs  = stringlist_get_size(hist_obs_keys);

    for(int hist_obs_nr = 0; hist_obs_nr < num_hist_obs; hist_obs_nr++)
    {
      const char               * sum_key       = stringlist_iget(hist_obs_keys, hist_obs_nr); 
      const conf_instance_type * hist_obs_conf = conf_instance_get_sub_instance_ref(enkf_conf, sum_key);
      summary_obs_type         * sum_obs       = summary_obs_alloc_from_HISTORY_OBSERVATION(hist_obs_conf, hist);

      obs_node_type            * obs_node      = obs_node_alloc(sum_obs, sum_key, sum_key, summary_obs , num_restarts, false,
                                                                summary_obs_get_observations__, summary_obs_measure__,
                                                                summary_obs_free__, NULL);

      /** This not exactly a sexy solution... obs_node should really just ask for this when it's needed. */
      for(int restart_nr = 0; restart_nr < num_restarts; restart_nr++)
      {
        if(!summary_obs_default_used(sum_obs, restart_nr))
          obs_node_activate_report_step(obs_node, restart_nr, restart_nr);
      }

      enkf_obs_add_obs(enkf_obs, sum_key, obs_node);
    }

    stringlist_free(hist_obs_keys);
  }



  /** Handle SUMMARY_OBSERVATION instances. */
  {
    stringlist_type * sum_obs_keys = conf_instance_alloc_list_of_sub_instances_of_class_by_name(enkf_conf, "SUMMARY_OBSERVATION");
    int               num_sum_obs  = stringlist_get_size(sum_obs_keys);

    for(int sum_obs_nr = 0; sum_obs_nr < num_sum_obs; sum_obs_nr++)
    {
      const char               * obs_key      = stringlist_iget(sum_obs_keys, sum_obs_nr);
      const conf_instance_type * sum_obs_conf = conf_instance_get_sub_instance_ref(enkf_conf, obs_key);
      summary_obs_type         * sum_obs      = summary_obs_alloc_from_SUMMARY_OBSERVATION(sum_obs_conf, hist);
      const char               * sum_key      = summary_obs_get_summary_key_ref(sum_obs);

      obs_node_type            * obs_node     = obs_node_alloc(sum_obs, sum_key, obs_key, summary_obs , num_restarts, false,
                                                               summary_obs_get_observations__, summary_obs_measure__,
                                                               summary_obs_free__, NULL);

      /** This not exactly a sexy solution... obs_node should really just ask for this when it's needed. */
      for(int restart_nr = 0; restart_nr < num_restarts; restart_nr++)
      {
        if(!summary_obs_default_used(sum_obs, restart_nr))
          obs_node_activate_report_step(obs_node, restart_nr, restart_nr);
      }

      enkf_obs_add_obs(enkf_obs, obs_key, obs_node);
    }

    stringlist_free(sum_obs_keys);
  }



  /** Handle BLOCK_OBSERVATION instances. */
  {
    stringlist_type * block_obs_keys = conf_instance_alloc_list_of_sub_instances_of_class_by_name(enkf_conf, "BLOCK_OBSERVATION");
    int               num_block_obs  = stringlist_get_size(block_obs_keys);

    for(int block_obs_nr = 0; block_obs_nr < num_block_obs; block_obs_nr++)
    {
      const char               * obs_key        = stringlist_iget(block_obs_keys, block_obs_nr);
      const conf_instance_type * block_obs_conf = conf_instance_get_sub_instance_ref(enkf_conf, obs_key);
      field_obs_type           * block_obs      = field_obs_alloc_from_BLOCK_OBSERVATION(block_obs_conf, hist);
      const char               * field_name     = field_obs_get_field_name_ref(block_obs);
      int                        restart_nr     = field_obs_get_restart_nr(block_obs);

      obs_node_type * obs_node                   = obs_node_alloc(block_obs, field_name, obs_key, field_obs , num_restarts, false,
                                                                  field_obs_get_observations__, field_obs_measure__,
                                                                  field_obs_free__, NULL);

      obs_node_activate_report_step(obs_node, restart_nr, restart_nr);

      enkf_obs_add_obs(enkf_obs, obs_key, obs_node);
    }


    stringlist_free(block_obs_keys);
  }



  conf_instance_free(enkf_conf      );
  conf_class_free(   enkf_conf_class);
  return enkf_obs;
}



static 
conf_class_type * enkf_obs_get_obs_conf_class(
)
{
  const char * enkf_conf_help = "An instance of the class ENKF_CONFIG shall contain neccessary infomation to run the enkf.";
  conf_class_type * enkf_conf_class = conf_class_alloc_empty("ENKF_CONFIG", true);
  conf_class_set_help(enkf_conf_class, enkf_conf_help);



  /** Create and insert HISTORY_OBSERVATION class. */
  {
    const char * help_class_history_observation = "The class HISTORY_OBSERVATION is used to condition on a time series from the production history. The name of the an instance is used to define the item to condition on, and should be in summary.x syntax. E.g., creating a HISTORY_OBSERVATION instance with name GOPR:P4 conditions on GOPR for group P4.";
    conf_class_type * history_observation_class = conf_class_alloc_empty("HISTORY_OBSERVATION", false);
    conf_class_set_help(history_observation_class, help_class_history_observation);

    const char * help_item_spec_error_mode = "The string ERROR_MODE gives the error mode for the observation.";
    conf_item_spec_type * item_spec_error_mode = conf_item_spec_alloc("ERROR_MODE", true, DT_STR);

    conf_item_spec_add_restriction(item_spec_error_mode, "REL");
    conf_item_spec_add_restriction(item_spec_error_mode, "ABS");
    conf_item_spec_add_restriction(item_spec_error_mode, "RELMIN");

    conf_item_spec_set_default_value(item_spec_error_mode, "RELMIN");
    conf_item_spec_set_help(item_spec_error_mode, help_item_spec_error_mode);

    const char * help_item_spec_error = "The positive floating number ERROR gives the standard deviation (ABS) or the relative uncertainty (REL/RELMIN) of the observations.";
    conf_item_spec_type * item_spec_error     = conf_item_spec_alloc("ERROR", true, DT_POSFLOAT);
    conf_item_spec_set_default_value(item_spec_error, "0.10");
    conf_item_spec_set_help(item_spec_error, help_item_spec_error);

    const char * help_item_spec_error_min = "The positive floating point number ERROR_MIN gives the minimum value for the standard deviation of the observation when RELMIN is used.";
    conf_item_spec_type * item_spec_error_min = conf_item_spec_alloc("ERROR_MIN", true, DT_POSFLOAT);
    conf_item_spec_set_default_value(item_spec_error_min, "0.10");
    conf_item_spec_set_help(item_spec_error_min, help_item_spec_error_min);

    conf_class_insert_owned_item_spec(history_observation_class, item_spec_error_mode);
    conf_class_insert_owned_item_spec(history_observation_class, item_spec_error);
    conf_class_insert_owned_item_spec(history_observation_class, item_spec_error_min);

    conf_class_insert_owned_sub_class(enkf_conf_class, history_observation_class);
  }



  /** Create and insert SUMMARY_OBSERVATION class. */
  {
    const char * help_class_summary_observation = "The class SUMMARY_OBSERVATION can be used to condition on any observation whos simulated value is written to the summary file.";
    conf_class_type * summary_observation_class = conf_class_alloc_empty("SUMMARY_OBSERVATION", false);
    conf_class_set_help(summary_observation_class, help_class_summary_observation);

    const char * help_item_spec_value = "The floating point number VALUE gives the observed value.";
    conf_item_spec_type * item_spec_value = conf_item_spec_alloc("VALUE", true, DT_FLOAT);
    conf_item_spec_set_help(item_spec_value, help_item_spec_value);
  

    const char * help_item_spec_error = "The positive floating point number ERROR is the standard deviation of the observed value.";
    conf_item_spec_type * item_spec_error = conf_item_spec_alloc("ERROR", true, DT_POSFLOAT);
    conf_item_spec_set_help(item_spec_error, help_item_spec_error);

    const char * help_item_spec_date = "The DATE item gives the observation time as the date date it occured. Format is dd/mm/yyyy.";
    conf_item_spec_type * item_spec_date = conf_item_spec_alloc("DATE", false, DT_DATE);
    conf_item_spec_set_help(item_spec_date, help_item_spec_date);

    const char * help_item_spec_days = "The DAYS item gives the observation time as days after simulation start.";
    conf_item_spec_type * item_spec_days = conf_item_spec_alloc("DAYS", false, DT_POSFLOAT);
    conf_item_spec_set_help(item_spec_days, help_item_spec_days);

    const char * help_item_spec_restart = "The RESTART item gives the observation time as the ECLIPSE restart nr.";
    conf_item_spec_type * item_spec_restart = conf_item_spec_alloc("RESTART", false, DT_POSINT);
    conf_item_spec_set_help(item_spec_restart, help_item_spec_restart);


    const char * help_item_spec_sumkey = "The string SUMMARY_KEY is used to look up the simulated value in the summary file. It has the same format as the summary.x program, e.g. WOPR:P4";
    conf_item_spec_type * item_spec_sumkey = conf_item_spec_alloc("KEY", true, DT_STR);
    conf_item_spec_set_help(item_spec_sumkey, help_item_spec_sumkey);

    conf_class_insert_owned_item_spec(summary_observation_class, item_spec_value);
    conf_class_insert_owned_item_spec(summary_observation_class, item_spec_error);
    conf_class_insert_owned_item_spec(summary_observation_class, item_spec_date);
    conf_class_insert_owned_item_spec(summary_observation_class, item_spec_days);
    conf_class_insert_owned_item_spec(summary_observation_class, item_spec_restart);
    conf_class_insert_owned_item_spec(summary_observation_class, item_spec_sumkey);

    /** Create a mutex on DATE, DAYS and RESTART. */
    conf_item_mutex_type * time_mutex = conf_item_mutex_alloc(true);
    conf_class_insert_owned_item_mutex(summary_observation_class, time_mutex);

    conf_item_mutex_add_item_spec(time_mutex, item_spec_date);
    conf_item_mutex_add_item_spec(time_mutex, item_spec_days);
    conf_item_mutex_add_item_spec(time_mutex, item_spec_restart);

    conf_class_insert_owned_sub_class(enkf_conf_class, summary_observation_class);
  }



  /** Create and insert BLOCK_OBSERVATION class. */
  {
    const char * help_class_block_observation = "The class BLOCK_OBSERVATION can be used to condition on an observation whos simulated values are block/cell values of a field, e.g. RFT tests.";
    conf_class_type * block_observation_class = conf_class_alloc_empty("BLOCK_OBSERVATION", false);
    conf_class_set_help(block_observation_class, help_class_block_observation);

    const char * help_item_spec_field = "The item FIELD gives the observed field. E.g., ECLIPSE fields such as PRESSURE, SGAS or any user defined fields such as PORO or PERMX.";
    conf_item_spec_type * item_spec_field = conf_item_spec_alloc("FIELD", true, DT_STR);
    conf_item_spec_set_help(item_spec_field, help_item_spec_field);

    const char * help_item_spec_date = "The DATE item gives the date of the observation. Format is dd/mm/yyyy.";
    conf_item_spec_type * item_spec_date = conf_item_spec_alloc("DATE", false, DT_DATE);
    conf_item_spec_set_help(item_spec_date, help_item_spec_date);

    const char * help_item_spec_days = "The DAYS item gives the observation time as days after simulation start.";
    conf_item_spec_type * item_spec_days = conf_item_spec_alloc("DAYS", false, DT_POSFLOAT);
    conf_item_spec_set_help(item_spec_days, help_item_spec_days);

    const char * help_item_spec_restart = "The RESTART item gives the observation time as the ECLIPSE restart nr.";
    conf_item_spec_type * item_spec_restart = conf_item_spec_alloc("RESTART", false, DT_POSINT);
    conf_item_spec_set_help(item_spec_restart, help_item_spec_restart);

    conf_class_insert_owned_item_spec(block_observation_class, item_spec_field);
    conf_class_insert_owned_item_spec(block_observation_class, item_spec_date);
    conf_class_insert_owned_item_spec(block_observation_class, item_spec_days);
    conf_class_insert_owned_item_spec(block_observation_class, item_spec_restart);

    /** Create a mutex on DATE, DAYS and RESTART. */
    conf_item_mutex_type * time_mutex = conf_item_mutex_alloc(true);
    conf_class_insert_owned_item_mutex(block_observation_class, time_mutex);

    conf_item_mutex_add_item_spec(time_mutex, item_spec_date);
    conf_item_mutex_add_item_spec(time_mutex, item_spec_days);
    conf_item_mutex_add_item_spec(time_mutex, item_spec_restart);

    /** Create and insert the sub class OBS. */
    {
      const char * help_class_obs = "The class OBS is used to specify a single observed point.";
      conf_class_type * obs_class = conf_class_alloc_empty("OBS", true);
      conf_class_set_help(obs_class, help_class_obs);

      const char * help_item_i = "The item I gives the I index of the block observation.";
      conf_item_spec_type * item_spec_i = conf_item_spec_alloc("I", true, DT_POSINT);
      conf_item_spec_set_help(item_spec_i, help_item_i);

      const char * help_item_j = "The item J gives the J index of the block observation.";
      conf_item_spec_type * item_spec_j = conf_item_spec_alloc("J", true, DT_POSINT);
      conf_item_spec_set_help(item_spec_j, help_item_j);

      const char * help_item_k = "The item K gives the K index of the block observation.";
      conf_item_spec_type * item_spec_k = conf_item_spec_alloc("K", true, DT_POSINT);
      conf_item_spec_set_help(item_spec_k, help_item_k);

      const char * help_item_spec_value = "The floating point number VALUE gives the observed value.";
      conf_item_spec_type * item_spec_value = conf_item_spec_alloc("VALUE", true, DT_FLOAT);
      conf_item_spec_set_help(item_spec_value, help_item_spec_value);
    
      const char * help_item_spec_error = "The positive floating point number ERROR is the standard deviation of the observed value.";
      conf_item_spec_type * item_spec_error = conf_item_spec_alloc("ERROR", true, DT_POSFLOAT);
      conf_item_spec_set_help(item_spec_error, help_item_spec_error);

      conf_class_insert_owned_item_spec(obs_class, item_spec_i);
      conf_class_insert_owned_item_spec(obs_class, item_spec_j);
      conf_class_insert_owned_item_spec(obs_class, item_spec_k);
      conf_class_insert_owned_item_spec(obs_class, item_spec_value);
      conf_class_insert_owned_item_spec(obs_class, item_spec_error);

      conf_class_insert_owned_sub_class(block_observation_class, obs_class);
    }

    conf_class_insert_owned_sub_class(enkf_conf_class, block_observation_class);
  }

  return enkf_conf_class;
}



/**
   Allocates a stringlist of obs keys which correspond to summary
   observations, these are then added to the state vector in
   enkf_main.
*/
stringlist_type * enkf_obs_alloc_summary_vars(
  enkf_obs_type * enkf_obs)
{
  stringlist_type * summary_vars = stringlist_alloc_new();
  char * key = hash_iter_get_first_key( enkf_obs->obs_hash );
  while ( key != NULL) {
    obs_node_type * obs_node = hash_get( enkf_obs->obs_hash , key);
    if (obs_node_get_impl_type(obs_node) == summary_obs) 
      stringlist_append_ref(summary_vars , obs_node_get_state_kw(obs_node));
    key = hash_iter_get_next_key( enkf_obs->obs_hash );
  }
  return summary_vars;
}

