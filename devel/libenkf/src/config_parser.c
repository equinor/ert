#include <conf.h>
static
conf_class_type * history_observation_class(
  void)
{
  const char * ho_help            = "The class HISTORY_OBSERVATION is used to condition on a time series from the production history. The name of the an instance is used to define the item to condition on, and should be in summary.x syntax.";
  const char * ho_error_mode_help = "The string ERROR_MODE gives the error mode for the observation.";
  const char * ho_error_help      = "The positive floating number ERROR gives the standard deviation (ABS) or the relative uncertainty (REL/RELMIN) of the observations.";
  const char * ho_error_min_help  = "The positive floating point number ERROR_MIN gives the minimum value for the standard deviation of the observation when ERROR_MODE is set to RELMIN.";

  conf_class_type * ho_class = conf_class_alloc_empty("HISTORY_OBSERVATION", false , false, ho_help);

  conf_item_spec_type * error_mode = conf_item_spec_alloc("ERROR_MODE", true, DT_STR , ho_error_mode_help );
  conf_item_spec_add_restriction(  error_mode, "REL");
  conf_item_spec_add_restriction(  error_mode, "ABS");
  conf_item_spec_add_restriction(  error_mode, "RELMIN");
  conf_item_spec_set_default_value(error_mode, "RELMIN");

  conf_item_spec_type * error = conf_item_spec_alloc("ERROR", true, DT_POSFLOAT , ho_error_help);
  conf_item_spec_set_default_value(error, "0.10");

  conf_item_spec_type * error_min = conf_item_spec_alloc("ERROR_MIN", true, DT_POSFLOAT , ho_error_min_help);
  conf_item_spec_set_default_value(error_min, "0.10");

  conf_class_insert_owned_item_spec(ho_class, error_mode);
  conf_class_insert_owned_item_spec(ho_class, error);
  conf_class_insert_owned_item_spec(ho_class, error_min);

  /** Sub class segment. */
  {
    const char * seg_help            = "The class SEGMENT is used to fine tune the error model.";
    const char * seg_start_help      = "The first restart in the segment.";
    const char * seg_stop_help       = "The last restart in the segment.";
    const char * seg_error_mode_help = "The string ERROR_MODE gives the error mode for the observation.";
    const char * seg_error_help      = "The positive floating number ERROR gives the standard deviation (ABS) or the relative uncertainty (REL/RELMIN) of the observations.";
    const char * seg_error_min_help  = "The positive floating point number ERROR_MIN gives the minimum value for the standard deviation of the observation when ERROR_MODE is set to RELMIN.";

    conf_class_type * seg_class = conf_class_alloc_empty("SEGMENT", false , false, seg_help);

    conf_item_spec_type * start_seg = conf_item_spec_alloc("START", true, DT_INT, seg_start_help);
    conf_item_spec_type * stop_seg  = conf_item_spec_alloc("STOP",  true, DT_INT, seg_stop_help);

    conf_item_spec_type * error_mode_seg = conf_item_spec_alloc("ERROR_MODE", true, DT_STR , seg_error_mode_help);
    conf_item_spec_add_restriction(  error_mode_seg, "REL");
    conf_item_spec_add_restriction(  error_mode_seg, "ABS");
    conf_item_spec_add_restriction(  error_mode_seg, "RELMIN");
    conf_item_spec_set_default_value(error_mode_seg, "RELMIN");

    conf_item_spec_type * error_seg = conf_item_spec_alloc("ERROR", true, DT_POSFLOAT , seg_error_help);
    conf_item_spec_set_default_value(error_seg, "0.10");

    conf_item_spec_type * error_min_seg = conf_item_spec_alloc("ERROR_MIN", true, DT_POSFLOAT ,  seg_error_min_help);
    conf_item_spec_set_default_value(error_min_seg, "0.10");


    conf_class_insert_owned_item_spec(seg_class, start_seg);
    conf_class_insert_owned_item_spec(seg_class, stop_seg);
    conf_class_insert_owned_item_spec(seg_class, error_mode_seg);
    conf_class_insert_owned_item_spec(seg_class, error_seg);
    conf_class_insert_owned_item_spec(seg_class, error_min_seg);

    conf_class_insert_owned_sub_class(ho_class, seg_class);
  }

  return ho_class; 
}



static
conf_class_type * summary_observation_class(
  void)
{
  const char * so_help         = "The class SUMMARY_OBSERVATION can be used to condition on any observation whos simulated value is written to the ECLIPSE summary file.";
  const char * so_value_help   = "The floating point number VALUE gives the observed value.";
  const char * so_error_help   = "The positive floating point number ERROR is the standard deviation of the observed value.";
  const char * so_date_help    = "The DATE item gives the observation time as the date date it occured. Format is dd/mm/yyyy.";
  const char * so_days_help    = "The DAYS item gives the observation time as days after simulation start.";
  const char * so_restart_help = "The RESTART item gives the observation time as the ECLIPSE restart nr.";
  const char * so_key_help     = "The string KEY is used to look up the simulated value in the summary file. It has the same format as the summary.x program, e.g. WOPR:P4";

  conf_class_type * so_class = conf_class_alloc_empty("SUMMARY_OBSERVATION", false , false, so_help);

  conf_item_spec_type * value   = conf_item_spec_alloc("VALUE"  , true , DT_FLOAT,    so_value_help);
  conf_item_spec_type * error   = conf_item_spec_alloc("ERROR"  , true , DT_POSFLOAT, so_error_help);
  conf_item_spec_type * date    = conf_item_spec_alloc("DATE"   , false, DT_DATE,     so_date_help);
  conf_item_spec_type * days    = conf_item_spec_alloc("DAYS"   , false, DT_POSFLOAT, so_days_help);
  conf_item_spec_type * restart = conf_item_spec_alloc("RESTART", false, DT_POSINT ,  so_restart_help);
  conf_item_spec_type * key     = conf_item_spec_alloc("KEY"    , true , DT_STR ,     so_key_help);

  conf_class_insert_owned_item_spec(so_class, value);
  conf_class_insert_owned_item_spec(so_class, error);
  conf_class_insert_owned_item_spec(so_class, date);
  conf_class_insert_owned_item_spec(so_class, days);
  conf_class_insert_owned_item_spec(so_class, restart);
  conf_class_insert_owned_item_spec(so_class, key);

  /** Create a mutex on DATE, DAYS and RESTART. */
  conf_item_mutex_type * time_mutex = conf_class_new_item_mutex(so_class , true , false);

  conf_item_mutex_add_item_spec(time_mutex, date);
  conf_item_mutex_add_item_spec(time_mutex, days);
  conf_item_mutex_add_item_spec(time_mutex, restart);

  return so_class;
}



static
conf_class_type * block_observation_class(
  void)
{
  const char * bo_help         = "The class BLOCK_OBSERVATION can be used to condition on an observation whos simulated values are block/cell values of a field, e.g. RFT tests.";
  const char * bo_field_help   = "The item FIELD gives the observed field. E.g., ECLIPSE fields such as PRESSURE, SGAS or any user defined fields such as PORO or PERMX.";
  const char * bo_date_help    = "The DATE item gives the observation time as the date date it occured. Format is dd/mm/yyyy.";
  const char * bo_days_help    = "The DAYS item gives the observation time as days after simulation start.";
  const char * bo_restart_help = "The RESTART item gives the observation time as the ECLIPSE restart nr.";

  conf_class_type * bo_class = conf_class_alloc_empty("BLOCK_OBSERVATION", false , false, bo_help);

  conf_item_spec_type * field   = conf_item_spec_alloc("FIELD"  , true , DT_STR     , bo_field_help);
  conf_item_spec_type * date    = conf_item_spec_alloc("DATE"   , false, DT_DATE    , bo_date_help);
  conf_item_spec_type * days    = conf_item_spec_alloc("DAYS"   , false, DT_POSFLOAT, bo_days_help);
  conf_item_spec_type * restart = conf_item_spec_alloc("RESTART", false, DT_POSINT  , bo_restart_help);
  
  conf_class_insert_owned_item_spec(bo_class, field);
  conf_class_insert_owned_item_spec(bo_class, date);
  conf_class_insert_owned_item_spec(bo_class, days);
  conf_class_insert_owned_item_spec(bo_class, restart);

  /** Create a mutex on DATE, DAYS and RESTART. */
  conf_item_mutex_type * time_mutex = conf_class_new_item_mutex(bo_class , true , false);
  conf_item_mutex_add_item_spec(time_mutex, date);
  conf_item_mutex_add_item_spec(time_mutex, days);
  conf_item_mutex_add_item_spec(time_mutex, restart);

  /** Create and insert the sub class OBS. */
  {
    const char * obs_help       = "The class OBS is used to specify a single observed point.";
    const char * obs_i_help     = "The item I gives the I index of the block observation.";
    const char * obs_j_help     = "The item J gives the J index of the block observation.";
    const char * obs_k_help     = "The item K gives the K index of the block observation.";
    const char * obs_value_help = "The floating point number VALUE gives the observed value.";
    const char * obs_error_help = "The positive floating point number ERROR is the standard deviation of the observed value.";

    conf_class_type * obs_class = conf_class_alloc_empty("OBS", true , false, obs_help);

    conf_item_spec_type * i     = conf_item_spec_alloc("I"    , true, DT_POSINT  , obs_i_help);
    conf_item_spec_type * j     = conf_item_spec_alloc("J"    , true, DT_POSINT  , obs_j_help);
    conf_item_spec_type * k     = conf_item_spec_alloc("K"    , true, DT_POSINT  , obs_k_help);
    conf_item_spec_type * value = conf_item_spec_alloc("VALUE", true, DT_FLOAT   , obs_value_help);
    conf_item_spec_type * error = conf_item_spec_alloc("ERROR", true, DT_POSFLOAT, obs_error_help);

    conf_class_insert_owned_item_spec(obs_class, i);
    conf_class_insert_owned_item_spec(obs_class, j);
    conf_class_insert_owned_item_spec(obs_class, k);
    conf_class_insert_owned_item_spec(obs_class, value);
    conf_class_insert_owned_item_spec(obs_class, error);

    conf_class_insert_owned_sub_class(bo_class, obs_class);
  }

  return bo_class;
}



static
conf_class_type * general_observation_class(
  void)
{
  const char * go_help            =  "The class general_observation is used for general observations";
  const char * go_restart_help    = "The RESTART item gives the observation time as the ECLIPSE restart nr.";
  const char * go_field_help      = "The item DATA gives the observed GEN_DATA instance.";
  const char * go_date_help       = "The DATE item gives the observation time as the date date it occured. Format is dd/mm/yyyy.";
  const char * go_days_help       = "The DAYS item gives the observation time as days after simulation start.";
  const char * go_obs_file_help   = "The name of an (ascii) file with observation values.";
  const char * go_value_help      = "One scalar observation value.";
  const char * go_error_help      = "One scalar observation error.";
  const char * go_index_list_help = "A list of indicies - possibly with ranges which should be observed in the target field.";
  const char * go_index_file_help = "An ASCII file containing a list of indices which should be observed in the target field.";

  conf_class_type * go_class = conf_class_alloc_empty("GENERAL_OBSERVATION" , false , false, go_help);

  conf_item_spec_type * field      = conf_item_spec_alloc("DATA"      , true , DT_STR     , go_field_help);
  conf_item_spec_type * date       = conf_item_spec_alloc("DATE"      , false, DT_DATE    , go_date_help);
  conf_item_spec_type * days       = conf_item_spec_alloc("DAYS"      , false, DT_POSFLOAT, go_days_help);
  conf_item_spec_type * restart    = conf_item_spec_alloc("RESTART"   , false, DT_POSINT  , go_restart_help);
  conf_item_spec_type * obs_file   = conf_item_spec_alloc("OBS_FILE"  , false, DT_FILE    , go_obs_file_help);
  conf_item_spec_type * value      = conf_item_spec_alloc("VALUE"     , false, DT_FLOAT   , go_value_help);
  conf_item_spec_type * error      = conf_item_spec_alloc("ERROR"     , false, DT_FLOAT   , go_error_help);
  conf_item_spec_type * index_list = conf_item_spec_alloc("INDEX_LIST", false, DT_STR     , go_index_list_help);
  conf_item_spec_type * index_file = conf_item_spec_alloc("INDEX_FILE", false, DT_FILE    , go_index_file_help);

  conf_class_insert_owned_item_spec(go_class, field);
  conf_class_insert_owned_item_spec(go_class, date);
  conf_class_insert_owned_item_spec(go_class, days);
  conf_class_insert_owned_item_spec(go_class, restart);
  conf_class_insert_owned_item_spec(go_class, obs_file);
  conf_class_insert_owned_item_spec(go_class, value);
  conf_class_insert_owned_item_spec(go_class, error);
  conf_class_insert_owned_item_spec(go_class, index_list);
  conf_class_insert_owned_item_spec(go_class, index_file);


  /** Create a mutex on DATE, DAYS and RESTART. */
  {
    conf_item_mutex_type * time_mutex = conf_class_new_item_mutex(go_class , true , false);
    
    conf_item_mutex_add_item_spec(time_mutex, date);
    conf_item_mutex_add_item_spec(time_mutex, days);
    conf_item_mutex_add_item_spec(time_mutex, restart);
  }
  
  /** Create a mutex on VALUE and OBS_FILE. */
  {
    conf_item_mutex_type * value_mutex = conf_class_new_item_mutex( go_class , true  , false);

    conf_item_mutex_add_item_spec( value_mutex, value);
    conf_item_mutex_add_item_spec( value_mutex, obs_file);
  }

  /**
    If the observation is in terms of VALUE - we must also have ERROR. 
    To achieve this, we use an inverse mutex.
  */
  {
    conf_item_mutex_type * value_error_mutex = conf_class_new_item_mutex( go_class , false , true);
    conf_item_mutex_add_item_spec( value_error_mutex, value);
    conf_item_mutex_add_item_spec( value_error_mutex, error);
  }

  /* 
     The default is that all the elements in DATA are observed, but
     we can restrict ourselves to a list of indices, with either the
     INDEX_LIST or INDEX_FILE keywords.
  */
  {
    conf_item_mutex_type * index_mutex = conf_class_new_item_mutex( go_class , false , false);
    
    conf_item_mutex_add_item_spec(index_mutex , index_list);
    conf_item_mutex_add_item_spec(index_mutex , index_file);
  }
    
  return go_class;
}
