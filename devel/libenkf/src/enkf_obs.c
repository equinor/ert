#include <string.h>
#include <stdlib.h>
#include <hash.h>
#include <util.h>
#include <conf.h>
#include <enkf_obs.h>
#include <summary_obs.h>
#include <field_obs.h>
#include <enkf_fs.h>
#include <obs_vector.h>
#include <msg.h>
#include <enkf_state.h>
#include <local_ministep.h>
#include <local_reportstep.h>
#include <local_config.h>

/*

The observation system
----------------------

The observation system in the EnKF code is a three leayer system. At
the the top is the enkf_obs_type. The enkf_main object contains one
enkf_obs instance which has internalized ALL the observation data. In
enkf_obs the the data is internalized in a hash table, where the keys
in the table are the keys used the observation file.

The next level is the obs_vector type which is a vector of length
num_report_steps. Each element in this vector can either point a
spesific observation instance (which actually contains the data), or
be NULL, if the observation is not active at this report step. In
addition the obs_vector contains function pointers to manipulate the
observation data at the lowest level.

At the lowest level we have spsesific observation instances,
field_obs, summary_obs and gen_obs. These instances contain the actual
data.

To summarize we can say:

  1. enkf_obs has ALL the observation data.

  2. obs_vector has the full time series for one observation key,
     i.e. all the watercuts in well P2.

  3. field_obs/gen_obs/summary_obs instances contain the actual
     observed data for one (logical) observation and one report step.


In the following example we have two observations

 WWCT:OP1 The water cut in well OP1. This is an observation which is
    active for many report steps, at the lowest level it is
    implemented as summary_obs.

 RFT_P2 This is an RFT test for one well. Only active at one report
    step, implemented at the lowest level as a field_obs instance.


 In the example below there are in total five report steps, hence all
 the obs_vector instances have five 'slots'. If there is no active
 observation for a particular report step, the corresponding pointer
 in the obs_vector instance is NULL.



      _____________________           _____________________
     / 		       	     enkf_obs                      \
     |							   |
     |							   |
     | obs_hash: {"WWCT:OP1" , "RFT_P2"}                   |
     |   	      |	          |                        |
     |  	      |	   	  |			   |
     \________________|___________|________________________/
		      |		  |
		      |		  |
		      |		  |
		      |		  \--------------------------------------------------------------\
		      |		  								 |
		      |										 |
		     \|/									 |
 |--- obs_vector: WWCT:OP1 -----------------------------------------------------|		 |
 | Function pointers:	    --------  --------	--------  --------  --------	|		 |
 | Pointing to the          |      |  |      |	|      |  |      |  |      |	|		 |
 | underlying               | NULL |  |  X   |	|  X   |  | NULL |  |  X   |	|		 |
 | implementation in the    |      |  |  |   |	|  |   |  |      |  |  |   |	|		 |
 | summary_obs object.      --------  ---|----	---|----  --------  ---|----	|		 |
 |---------------------------------------|---------|-------------------|--------|		 |
			       		 |	   |		       |			 |
			       		\|/	   |		       |			 |
                                |-- summary_obs -| |		      \|/			 |
                                | Value: 0.56..	 | |	       |-- summary_obs -|		 |
			       	| std  : 0.15..	 | |	       | Value: 0.70..  |		 |
			       	|----------------| |	       | std  : 0.25..  |		 |
			   			   |	       |----------------|		 |
  			   			  \|/						 |
                                          |-- summary_obs -|					 |
			   		  | Value: 0.62..  |					 |
			   		  | std  : 0.12..  |					 |
			   		  |----------------|					 |
												 |
												 |
												 |
  The observation WWCT:OP1 is an observation of summary type, and the				 |
  obs_vector conatins pointers to summary_obs instances; along iwth				 |
  function pointers to manipulate the summary_obs instances. The				 |
  observation is not active for report steps 0 and 3, so for these				 |
  report steps the obse vector has a NULL pointer.						 |
												 |
												 |
												 |
												 |
												 |
												 |
 |--- obs_vector: RFT_P2 -------------------------------------------------------|		 |
 | Function pointers:	    --------  --------	--------  --------  --------	|		 |
 | Pointing to the          |      |  |      |	|      |  |      |  |      |	|<---------------/
 | underlying               | NULL |  | NULL | 	| NULL |  |  X   |  | NULL |	|
 | implementation in the    |      |  |      | 	|      |  |  |   |  |      |   	|
 | field_obs object.        --------  --------	--------  ---|----  --------	|
 |-----------------------------------------------------------|------------------|
			 				     |
			 				     |
							    \|/
                                        |-- field_obs -----------------------------------|
                                        | i = 25 , j = 16, k = 10, value = 278, std = 10 |
                                        | i = 25 , j = 16, k = 11, value = 279, std = 10 |
                                        | i = 25 , j = 16, k = 12, value = 279, std = 10 |
                                        | i = 25 , j = 17, k = 12, value = 281, std = 10 |
                                        | i = 25 , j = 18, k = 12, value = 282, std = 10 |
				       	|------------------------------------------------|


 The observation RFT_P2 is an RFT observation which is only active at
 one report step, i.e. 4/5 pointers in the obs_vector are just NULL
 pointers. The oactual observation(s) are stored in a field_obs
 instance. 

 */


/**
TODO

    This static function header shall be removed when the
    configuration is unified....
*/
static conf_class_type * enkf_obs_get_obs_conf_class();



//////////////////////////////////////////////////////////////////////////////////////



struct enkf_obs_struct {
  /** A hash of obs_vector_types indexed by user provided keys. */
  hash_type * obs_hash;
};



//////////////////////////////////////////////////////////////////////////////////////



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




void enkf_obs_add_obs_vector(
  enkf_obs_type       * enkf_obs,
  const char          * key ,
  const obs_vector_type * vector)
{
  if (hash_has_key(enkf_obs->obs_hash , key))
    util_abort("%s: Observation with key:%s already added.\n",__func__ , key);
  hash_insert_hash_owned_ref(enkf_obs->obs_hash , key , vector , obs_vector_free__);
}


bool enkf_obs_has_key(const enkf_obs_type * obs , const char * key) {
  return hash_has_key(obs->obs_hash , key);
}

obs_vector_type * enkf_obs_get_vector(const enkf_obs_type * obs, const char * key) {
  return hash_get(obs->obs_hash , key);
}



/*
  This will append observations and simulated responses 
  from report_step to obs_data and meas_matrix.
  Call obs_data_reset and meas_matrix_reset on
  obs_data and meas_matrix if you want to use fresh
  instances.
*/
void enkf_obs_get_obs_and_measure(
        const enkf_obs_type    * enkf_obs,
        enkf_fs_type           * fs,
        int                      report_step,
        state_enum               state,
        int                      ens_size,
        const enkf_state_type ** ensemble ,
        meas_matrix_type       * meas_matrix,
        obs_data_type          * obs_data,
	const local_ministep_type  * mstep) {
  
  hash_iter_type * iter = local_ministep_alloc_obs_iter( mstep );
  obs_data_reset( obs_data );
  meas_matrix_reset( meas_matrix );

  while ( !hash_iter_is_complete(iter) ) {
    const char * obs_key         = hash_iter_get_next_key( iter );
    obs_vector_type * obs_vector = hash_get( enkf_obs->obs_hash , obs_key );
    if (obs_vector_iget_active(obs_vector , report_step)) {                            /* The observation is active for this report step.     */
      const active_list_type * active_list = local_ministep_get_obs_active_list( mstep , obs_key );
      obs_vector_iget_observations(obs_vector , report_step , obs_data , active_list); /* Collect the observed data in the obs_data instance. */
      {
	int iens;
	for (iens = 0; iens < ens_size; iens++) {
	  enkf_node_type * enkf_node = enkf_state_get_node(ensemble[iens] , obs_vector_get_state_kw(obs_vector));
	  meas_vector_type * meas_vector = meas_matrix_iget_vector(meas_matrix , iens);
	  
	  enkf_fs_fread_node(fs , enkf_node , report_step , iens , state);
	  obs_vector_measure(obs_vector , report_step , enkf_node , meas_vector , active_list);
	}
      }
    }
  }
  hash_iter_free(iter);
}



/**
    TODO

    When configuration has been unified, this should take a
    conf_instance_type, not a config_file.
*/

enkf_obs_type * enkf_obs_fscanf_alloc(const char * config_file,  const history_type * hist, ensemble_config_type * ensemble_config) {
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

      obs_vector_type * obs_vector = obs_vector_alloc_from_HISTORY_OBSERVATION(hist_obs_conf , hist , ensemble_config);
      enkf_obs_add_obs_vector(enkf_obs, sum_key, obs_vector);
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
      obs_vector_type * obs_vector = obs_vector_alloc_from_SUMMARY_OBSERVATION(sum_obs_conf , hist , ensemble_config);
      enkf_obs_add_obs_vector(enkf_obs, obs_key, obs_vector);
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

      obs_vector_type * obs_vector = obs_vector_alloc_from_BLOCK_OBSERVATION(block_obs_conf , hist ,   ensemble_config);
      enkf_obs_add_obs_vector(enkf_obs, obs_key, obs_vector);
    }
    stringlist_free(block_obs_keys);
  }


  /** Handle GENERAL_OBSERVATION instances. */
  {
    stringlist_type * block_obs_keys = conf_instance_alloc_list_of_sub_instances_of_class_by_name(enkf_conf, "GENERAL_OBSERVATION");
    int               num_block_obs  = stringlist_get_size(block_obs_keys);
    
    for(int block_obs_nr = 0; block_obs_nr < num_block_obs; block_obs_nr++)
    {
      const char               * obs_key        = stringlist_iget(block_obs_keys, block_obs_nr);
      const conf_instance_type * gen_obs_conf   = conf_instance_get_sub_instance_ref(enkf_conf, obs_key);
  
      obs_vector_type * obs_vector = obs_vector_alloc_from_GENERAL_OBSERVATION(gen_obs_conf , hist  , ensemble_config);
      enkf_obs_add_obs_vector(enkf_obs, obs_key, obs_vector);
    }
    stringlist_free(block_obs_keys);
  }


  conf_instance_free(enkf_conf      );
  conf_class_free(   enkf_conf_class);

  return enkf_obs;
}



 static conf_class_type * enkf_obs_get_obs_conf_class( void ) {
  const char * enkf_conf_help = "An instance of the class ENKF_CONFIG shall contain neccessary infomation to run the enkf.";
  conf_class_type * enkf_conf_class = conf_class_alloc_empty("ENKF_CONFIG", true , false , enkf_conf_help);
  conf_class_set_help(enkf_conf_class, enkf_conf_help);



  /** Create and insert HISTORY_OBSERVATION class. */
  {
    const char * help_class_history_observation = "The class HISTORY_OBSERVATION is used to condition on a time series from the production history. The name of the an instance is used to define the item to condition on, and should be in summary.x syntax. E.g., creating a HISTORY_OBSERVATION instance with name GOPR:P4 conditions on GOPR for group P4.";

    conf_class_type * history_observation_class = conf_class_alloc_empty("HISTORY_OBSERVATION", false , false, help_class_history_observation);

    conf_item_spec_type * item_spec_error_mode = conf_item_spec_alloc("ERROR_MODE", true, true, DT_STR , "The string ERROR_MODE gives the error mode for the observation.");

    conf_item_spec_add_restriction(item_spec_error_mode, "REL");
    conf_item_spec_add_restriction(item_spec_error_mode, "ABS");
    conf_item_spec_add_restriction(item_spec_error_mode, "RELMIN");
    conf_item_spec_set_default_value(item_spec_error_mode, "RELMIN");

    conf_item_spec_type * item_spec_error     = conf_item_spec_alloc("ERROR", true, true, DT_POSFLOAT , "The positive floating number ERROR gives the standard deviation (ABS) or the relative uncertainty (REL/RELMIN) of the observations.");
    conf_item_spec_set_default_value(item_spec_error, "0.10");

    conf_item_spec_type * item_spec_error_min = conf_item_spec_alloc("ERROR_MIN", true, true, DT_POSFLOAT , "The positive floating point number ERROR_MIN gives the minimum value for the standard deviation of the observation when RELMIN is used.");
    conf_item_spec_set_default_value(item_spec_error_min, "0.10");

    conf_class_insert_owned_item_spec(history_observation_class, item_spec_error_mode);
    conf_class_insert_owned_item_spec(history_observation_class, item_spec_error);
    conf_class_insert_owned_item_spec(history_observation_class, item_spec_error_min);

    /** Sub class segment. */
    {
      const char * help_class_segment = "The class SEGMENT is used to fine tune the error model.";
      conf_class_type * segment_class = conf_class_alloc_empty("SEGMENT", false , false, help_class_segment);

      conf_item_spec_type * item_spec_start_segment = conf_item_spec_alloc("START", true, true, DT_INT, "The first restart in the segment.");
      conf_item_spec_type * item_spec_stop_segment  = conf_item_spec_alloc("STOP", true, true, DT_INT, "The last restart in the segment.");

      conf_item_spec_type * item_spec_error_mode_segment = conf_item_spec_alloc("ERROR_MODE", true, true, DT_STR , "The string ERROR_MODE gives the error mode for the observation.");
      conf_item_spec_add_restriction(item_spec_error_mode_segment, "REL");
      conf_item_spec_add_restriction(item_spec_error_mode_segment, "ABS");
      conf_item_spec_add_restriction(item_spec_error_mode_segment, "RELMIN");
      conf_item_spec_set_default_value(item_spec_error_mode_segment, "RELMIN");

      conf_item_spec_type * item_spec_error_segment     = conf_item_spec_alloc("ERROR", true, true, DT_POSFLOAT , "The positive floating number ERROR gives the standard deviation (ABS) or the relative uncertainty (REL/RELMIN) of the observations.");
      conf_item_spec_set_default_value(item_spec_error_segment, "0.10");

      conf_item_spec_type * item_spec_error_min_segment = conf_item_spec_alloc("ERROR_MIN", true, true, DT_POSFLOAT , "The positive floating point number ERROR_MIN gives the minimum value for the standard deviation of the observation when RELMIN is used.");
      conf_item_spec_set_default_value(item_spec_error_min_segment, "0.10");

  
      conf_class_insert_owned_item_spec(segment_class, item_spec_start_segment);
      conf_class_insert_owned_item_spec(segment_class, item_spec_stop_segment);
      conf_class_insert_owned_item_spec(segment_class, item_spec_error_mode_segment);
      conf_class_insert_owned_item_spec(segment_class, item_spec_error_segment);
      conf_class_insert_owned_item_spec(segment_class, item_spec_error_min_segment);

      conf_class_insert_owned_sub_class(history_observation_class, segment_class);
    }

    conf_class_insert_owned_sub_class(enkf_conf_class, history_observation_class);
  }



  /** Create and insert SUMMARY_OBSERVATION class. */
  {
    const char * help_class_summary_observation = "The class SUMMARY_OBSERVATION can be used to condition on any observation whos simulated value is written to the summary file.";
    conf_class_type * summary_observation_class = conf_class_alloc_empty("SUMMARY_OBSERVATION", false , false, help_class_summary_observation);

    const char * help_item_spec_value = "The floating point number VALUE gives the observed value.";
    conf_item_spec_type * item_spec_value = conf_item_spec_alloc("VALUE", true, true, DT_FLOAT , help_item_spec_value);


    const char * help_item_spec_error = "The positive floating point number ERROR is the standard deviation of the observed value.";
    conf_item_spec_type * item_spec_error = conf_item_spec_alloc("ERROR", true, true, DT_POSFLOAT ,help_item_spec_error );

    const char * help_item_spec_date = "The DATE item gives the observation time as the date date it occured. Format is dd/mm/yyyy.";
    conf_item_spec_type * item_spec_date = conf_item_spec_alloc("DATE", false, true, DT_DATE , help_item_spec_date);

    const char * help_item_spec_days = "The DAYS item gives the observation time as days after simulation start.";
    conf_item_spec_type * item_spec_days = conf_item_spec_alloc("DAYS", false, true, DT_POSFLOAT , help_item_spec_days);

    const char * help_item_spec_restart = "The RESTART item gives the observation time as the ECLIPSE restart nr.";
    conf_item_spec_type * item_spec_restart = conf_item_spec_alloc("RESTART", false, true, DT_POSINT , help_item_spec_restart);
    

    const char * help_item_spec_sumkey = "The string SUMMARY_KEY is used to look up the simulated value in the summary file. It has the same format as the summary.x program, e.g. WOPR:P4";
    conf_item_spec_type * item_spec_sumkey = conf_item_spec_alloc("KEY", true, true, DT_STR , help_item_spec_sumkey);

    conf_class_insert_owned_item_spec(summary_observation_class, item_spec_value);
    conf_class_insert_owned_item_spec(summary_observation_class, item_spec_error);
    conf_class_insert_owned_item_spec(summary_observation_class, item_spec_date);
    conf_class_insert_owned_item_spec(summary_observation_class, item_spec_days);
    conf_class_insert_owned_item_spec(summary_observation_class, item_spec_restart);
    conf_class_insert_owned_item_spec(summary_observation_class, item_spec_sumkey);

    /** Create a mutex on DATE, DAYS and RESTART. */
    conf_item_mutex_type * time_mutex = conf_class_new_item_mutex(summary_observation_class , true , false);

    conf_item_mutex_add_item_spec(time_mutex, item_spec_date);
    conf_item_mutex_add_item_spec(time_mutex, item_spec_days);
    conf_item_mutex_add_item_spec(time_mutex, item_spec_restart);

    conf_class_insert_owned_sub_class(enkf_conf_class, summary_observation_class);
  }



  /** Create and insert BLOCK_OBSERVATION class. */
  {
    const char * help_class_block_observation = "The class BLOCK_OBSERVATION can be used to condition on an observation whos simulated values are block/cell values of a field, e.g. RFT tests.";
    conf_class_type * block_observation_class = conf_class_alloc_empty("BLOCK_OBSERVATION", false , false, help_class_block_observation);

    const char * help_item_spec_field = "The item FIELD gives the observed field. E.g., ECLIPSE fields such as PRESSURE, SGAS or any user defined fields such as PORO or PERMX.";
    conf_item_spec_type * item_spec_field = conf_item_spec_alloc("FIELD", true, true, DT_STR , help_item_spec_field);

    const char * help_item_spec_date = "The DATE item gives the observation time as the date date it occured. Format is dd/mm/yyyy.";
    conf_item_spec_type * item_spec_date = conf_item_spec_alloc("DATE", false, true, DT_DATE , help_item_spec_date);

    const char * help_item_spec_days = "The DAYS item gives the observation time as days after simulation start.";
    conf_item_spec_type * item_spec_days = conf_item_spec_alloc("DAYS", false, true, DT_POSFLOAT , help_item_spec_days);

    const char * help_item_spec_restart = "The RESTART item gives the observation time as the ECLIPSE restart nr.";
    conf_item_spec_type * item_spec_restart = conf_item_spec_alloc("RESTART", false, true, DT_POSINT , help_item_spec_restart);
    
    
    conf_class_insert_owned_item_spec(block_observation_class, item_spec_field);
    conf_class_insert_owned_item_spec(block_observation_class, item_spec_date);
    conf_class_insert_owned_item_spec(block_observation_class, item_spec_days);
    conf_class_insert_owned_item_spec(block_observation_class, item_spec_restart);

    /** Create a mutex on DATE, DAYS and RESTART. */
    conf_item_mutex_type * time_mutex = conf_class_new_item_mutex(block_observation_class , true , false);
    conf_item_mutex_add_item_spec(time_mutex, item_spec_date);
    conf_item_mutex_add_item_spec(time_mutex, item_spec_days);
    conf_item_mutex_add_item_spec(time_mutex, item_spec_restart);

    /** Create and insert the sub class OBS. */
    {
      const char * help_class_obs = "The class OBS is used to specify a single observed point.";
      conf_class_type * obs_class = conf_class_alloc_empty("OBS", true , false, help_class_obs);

      const char * help_item_i = "The item I gives the I index of the block observation.";
      conf_item_spec_type * item_spec_i = conf_item_spec_alloc("I", true, true, DT_POSINT , help_item_i);

      const char * help_item_j = "The item J gives the J index of the block observation.";
      conf_item_spec_type * item_spec_j = conf_item_spec_alloc("J", true, true, DT_POSINT, help_item_j);

      const char * help_item_k = "The item K gives the K index of the block observation.";
      conf_item_spec_type * item_spec_k = conf_item_spec_alloc("K", true, true, DT_POSINT, help_item_k);

      const char * help_item_spec_value = "The floating point number VALUE gives the observed value.";
      conf_item_spec_type * item_spec_value = conf_item_spec_alloc("VALUE", true, true, DT_FLOAT , help_item_spec_value);

      const char * help_item_spec_error = "The positive floating point number ERROR is the standard deviation of the observed value.";
      conf_item_spec_type * item_spec_error = conf_item_spec_alloc("ERROR", true, true, DT_POSFLOAT , help_item_spec_error);

      conf_class_insert_owned_item_spec(obs_class, item_spec_i);
      conf_class_insert_owned_item_spec(obs_class, item_spec_j);
      conf_class_insert_owned_item_spec(obs_class, item_spec_k);
      conf_class_insert_owned_item_spec(obs_class, item_spec_value);
      conf_class_insert_owned_item_spec(obs_class, item_spec_error);

      conf_class_insert_owned_sub_class(block_observation_class, obs_class);
    }

    conf_class_insert_owned_sub_class(enkf_conf_class, block_observation_class);
  }

  /** Create and insert class for general observations. */
  {
    const char * help_item_spec_restart = "The RESTART item gives the observation time as the ECLIPSE restart nr.";
    const char * help_item_spec_field = "The item DATA gives the observed GEN_DATA instance.";
    const char * help_item_spec_date = "The DATE item gives the observation time as the date date it occured. Format is dd/mm/yyyy.";
    const char * help_item_spec_days = "The DAYS item gives the observation time as days after simulation start.";
  
    conf_class_type * gen_obs_class = conf_class_alloc_empty("GENERAL_OBSERVATION" , false , false, "The class general_observation is used for general observations");

    conf_item_spec_type * item_spec_field   = conf_item_spec_alloc("DATA", true, true, DT_STR , help_item_spec_field);
    conf_item_spec_type * item_spec_date    = conf_item_spec_alloc("DATE", false, true, DT_DATE , help_item_spec_date);
    conf_item_spec_type * item_spec_days    = conf_item_spec_alloc("DAYS", false, true, DT_POSFLOAT , help_item_spec_days);
    conf_item_spec_type * item_spec_restart = conf_item_spec_alloc("RESTART", false, true, DT_POSINT , help_item_spec_restart);

    conf_class_insert_owned_item_spec(gen_obs_class, item_spec_field);
    conf_class_insert_owned_item_spec(gen_obs_class, item_spec_date);
    conf_class_insert_owned_item_spec(gen_obs_class, item_spec_days);
    conf_class_insert_owned_item_spec(gen_obs_class, item_spec_restart);
    /** Create a mutex on DATE, DAYS and RESTART. */
    {
      conf_item_mutex_type * time_mutex = conf_class_new_item_mutex(gen_obs_class , true , false);
      
      conf_item_mutex_add_item_spec(time_mutex, item_spec_date);
      conf_item_mutex_add_item_spec(time_mutex, item_spec_days);
      conf_item_mutex_add_item_spec(time_mutex, item_spec_restart);
    }
    
    {
      conf_item_spec_type * item_spec_obs_file = conf_item_spec_alloc("OBS_FILE" , false , true, DT_FILE , "The name of an (ascii) file with observation values.");
      conf_item_spec_type * item_spec_value    = conf_item_spec_alloc("VALUE" , false , true, DT_FLOAT , "One scalar observation value.");
      conf_item_spec_type * item_spec_error    = conf_item_spec_alloc("ERROR" , false , true, DT_FLOAT , "One scalar observation error.");
      conf_item_mutex_type * value_mutex       = conf_class_new_item_mutex( gen_obs_class , true  , false);
      conf_item_mutex_type * value_error_mutex = conf_class_new_item_mutex( gen_obs_class , false , true);

      conf_class_insert_owned_item_spec(gen_obs_class , item_spec_obs_file);
      conf_class_insert_owned_item_spec(gen_obs_class , item_spec_value);
      conf_class_insert_owned_item_spec(gen_obs_class , item_spec_error);


      /* If the observation is in terms of VALUE - we must also have ERROR.
	 The conf system does not (currently ??) enforce this dependency. */
	 
      conf_item_mutex_add_item_spec( value_mutex , item_spec_value);
      conf_item_mutex_add_item_spec( value_mutex , item_spec_obs_file);

      conf_item_mutex_add_item_spec( value_error_mutex , item_spec_value);
      conf_item_mutex_add_item_spec( value_error_mutex , item_spec_error);
    }
  
  
    /* 
       The default is that all the elements in DATA are observed, but
       we can restrict ourselves to a list of indices, with either the
       INDEX_LIST or INDEX_FILE keywords.
    */
    {
      conf_item_spec_type * item_spec_index_list = conf_item_spec_alloc("INDEX_LIST" , false , true, DT_STR  , "A list of indicies - possibly with ranges which should be observed in the target field.");
      conf_item_spec_type * item_spec_index_file = conf_item_spec_alloc("INDEX_FILE" , false , true, DT_FILE , "An ASCII file containing a list of indices which should be observed in the target field.");
      conf_item_mutex_type * index_mutex         = conf_class_new_item_mutex( gen_obs_class , false , false);
      
      conf_class_insert_owned_item_spec(gen_obs_class, item_spec_index_list);
      conf_class_insert_owned_item_spec(gen_obs_class, item_spec_index_file);
      conf_item_mutex_add_item_spec(index_mutex , item_spec_index_list);
      conf_item_mutex_add_item_spec(index_mutex , item_spec_index_file);
    }
    
    conf_class_insert_owned_sub_class(enkf_conf_class, gen_obs_class);
  }


  return enkf_conf_class;
}



/**
   Allocates a stringlist of obs target keys which correspond to
   summary observations, these are then added to the state vector in
   enkf_main.
*/
stringlist_type * enkf_obs_alloc_typed_keylist(enkf_obs_type * enkf_obs , obs_impl_type obs_type) {
  stringlist_type * vars = stringlist_alloc_new();
  hash_iter_type  * iter = hash_iter_alloc(enkf_obs->obs_hash); 
  const char * key = hash_iter_get_next_key(iter);
  while ( key != NULL) {
    obs_vector_type * obs_vector = hash_get( enkf_obs->obs_hash , key);
    if (obs_vector_get_impl_type(obs_vector) == obs_type)
      stringlist_append_copy(vars , key);
    key = hash_iter_get_next_key(iter);
  }
  hash_iter_free(iter);
  return vars;
}



/**
   This function allocates a hash table which looks like this:

     {"OBS_KEY1": "STATE_KEY1", "OBS_KEY2": "STATE_KEY2", "OBS_KEY3": "STATE_KEY3", ....}

   where "OBS_KEY" represents the keys in the enkf_obs hash, and the
   value they are pointing at are the enkf_state keywords they are
   measuring. For instance if we have an observation with key "RFT_1A"
   the entry in the table will be:  ... "RFT_1A":  "PRESSURE", ..
   since an RFT observation observes the pressure.

   Let us consider the watercut in a well. Then the state_kw will
   typically be WWCT:P1 for a well named 'P1'. Let us assume that this
   well is observed both as a normal HISTORY observation from
   SCHEDULE, and from two separator tests, called S1 and S2. Then the
   hash table will look like this:

       "WWCT:P1": "WWCT:P1",
       "S1"     : "WWCT:P1",
       "S2"     : "WWCT:P1"


   I.e. there are three different observations keys, all observing the
   same state_kw.
*/



hash_type * enkf_obs_alloc_summary_map(enkf_obs_type * enkf_obs)
{
  hash_type      * map = hash_alloc();
  hash_iter_type * iter = hash_iter_alloc(enkf_obs->obs_hash); 
  const char * key = hash_iter_get_next_key(iter);
  while ( key != NULL) {
    obs_vector_type * obs_vector = hash_get( enkf_obs->obs_hash , key);
    hash_insert_ref( map , key , obs_vector_get_state_kw(obs_vector));
    key = hash_iter_get_next_key(iter);
  }
  return map;
}


hash_iter_type * enkf_obs_alloc_iter( const enkf_obs_type * enkf_obs ) {
  return hash_iter_alloc(enkf_obs->obs_hash);
}



/**
   This function takes a string like this: "PRESSURE:1,4,7" - it
   splits the string on ":" and tries to lookup a config object with
   that key. For the general string A:B:C:D it will try consecutively
   the keys: A, A:B, A:B:C, A:B:C:D. If a config object is found it is
   returned, otherwise NULL is returned.

   The last argument is the pointer to a string which will be updated
   with the node-spesific part of the full key. So for instance with
   the example "PRESSURE:1,4,7", the index_key will contain
   "1,4,7". If the full full_key is used to find an object index_key
   will be NULL, that also applies if no object is found.
*/



const obs_vector_type * enkf_obs_user_get_vector(const enkf_obs_type * obs , const char  * full_key, char ** index_key ) {
  const obs_vector_type * vector = NULL;
  char ** key_list;
  int     keys;
  int     key_length = 1;
  int offset;

  *index_key = NULL;
  util_split_string(full_key , ":" , &keys , &key_list);
  while (vector == NULL && key_length <= keys) {
    char * current_key = util_alloc_joined_string( (const char **) key_list , key_length , ":");
    if (enkf_obs_has_key(obs , current_key))
      vector = enkf_obs_get_vector(obs , current_key);
    else
      key_length++;
    offset = strlen( current_key );
    free( current_key );
  }
  if (vector != NULL) {
    if (offset < strlen( full_key ))
      *index_key = util_alloc_string_copy(&full_key[offset+1]);
  }

  util_free_stringlist(key_list , keys);
  return vector;
}



/*****************************************************************/


void enkf_obs_total_ensemble_chi2(const enkf_obs_type * obs , enkf_fs_type * fs , int ens_size , state_enum load_state , double * chi2) {
  const char      * obs_key;
  int iens;
  double          * obs_chi2 = util_malloc( ens_size * sizeof * obs_chi2 , __func__);
  
  for (iens = 0; iens < ens_size; iens++)
    chi2[iens] = 0;
  
  hash_iter_type * iter = hash_iter_alloc(obs->obs_hash);
  obs_key = hash_iter_get_next_key(iter);
  while (obs_key != NULL) {
    obs_vector_type * obs_vector = hash_get(obs->obs_hash , obs_key);

    obs_vector_ensemble_total_chi2( obs_vector , fs , ens_size , load_state , obs_chi2);
    for (iens = 0; iens < ens_size; iens++)
      chi2[iens] += obs_chi2[iens];
    
    obs_key = hash_iter_get_next_key(iter);
  }
  hash_iter_free(iter);
  free(obs_chi2);
}



void enkf_obs_ensemble_chi2(const enkf_obs_type * obs , enkf_fs_type * fs , int report_step , int ens_size , state_enum load_state , double * chi2) {
  const char      * obs_key;
  int iens;
  double          * obs_chi2 = util_malloc( ens_size * sizeof * obs_chi2 , __func__);

  for (iens = 0; iens < ens_size; iens++)
    chi2[iens] = 0;
    
  hash_iter_type * iter = hash_iter_alloc(obs->obs_hash);
  obs_key = hash_iter_get_next_key(iter);
  while (obs_key != NULL) {
    obs_vector_type * obs_vector = hash_get(obs->obs_hash , obs_key);

    obs_vector_ensemble_chi2( obs_vector , fs , report_step , ens_size , load_state , obs_chi2);
    for (iens = 0; iens < ens_size; iens++)
      chi2[iens] += obs_chi2[iens];
    
    obs_key = hash_iter_get_next_key(iter);
  }
  free(obs_chi2);
  hash_iter_free(iter);
}
