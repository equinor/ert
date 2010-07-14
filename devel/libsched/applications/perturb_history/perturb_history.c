#include <sched_file.h>
#include <sched_history.h>
#include <sched_kw.h>
#include <sched_kw_wconhist.h>
#include <ecl_util.h>
#include <util.h>
#include <stdlib.h>
#include <stdio.h>
#include <time_t_vector.h>
#include <double_vector.h>
#include <path_fmt.h>
#include <math.h>
#include <pert_util.h>
#include <well_rate.h>
#include <group_rate.h>
#include <config.h>
#include <sched_types.h>
#include <msg.h>

/*****************************************************************/


void perturb_wconhist( void * void_kw , int restart_nr , void * arg) {
  sched_kw_wconhist_type * kw = sched_kw_wconhist_safe_cast( void_kw );
  {
    hash_type * group_hash = hash_safe_cast( arg );
    hash_iter_type * group_iter = hash_iter_alloc( group_hash );
    while (!hash_iter_is_complete( group_iter )) {
      group_rate_type * group_rate = hash_iter_get_next_value( group_iter );
      if (group_rate_is_producer( group_rate ))
        group_rate_update_wconhist( group_rate , kw , restart_nr );
    }
  }
}



void perturb_wconinje( void * void_kw , int restart_nr , void * arg) {
  sched_kw_wconinje_type * kw = sched_kw_wconinje_safe_cast( void_kw );
  {
    hash_type * group_hash = hash_safe_cast( arg );
    hash_iter_type * group_iter = hash_iter_alloc( group_hash );
    while (!hash_iter_is_complete( group_iter )) {
      group_rate_type * group_rate = hash_iter_get_next_value( group_iter );
      if (!group_rate_is_producer( group_rate ))
        group_rate_update_wconinje( group_rate , kw , restart_nr );
    }
  }
}




void config_init(config_type * config ) {
  config_item_type * item;

  config_add_key_value(config , "NUM_REALIZATIONS" , true , CONFIG_INT );
  config_add_key_value(config , "SCHEDULE_FILE"    , true , CONFIG_EXISTING_FILE);
  config_add_key_value(config , "DATA_FILE"        , true , CONFIG_EXISTING_FILE);
  config_add_key_value(config , "TARGET"           , true , CONFIG_STRING );
  


  item = config_add_item( config , "GROUP_RATE" , false , true );  /* Group name as part of parsing */
  config_item_set_argc_minmax(item , 4 , 4 , (const config_item_types[4]) {   CONFIG_STRING,           /* Group name */
                                                                              CONFIG_STRING ,          /* Phase */
                                                                              CONFIG_STRING ,          /* PRODUCER / INJECTOR */
                                                                              CONFIG_EXISTING_FILE});  /* File with min / max shift */
  config_item_set_indexed_selection_set( item , 1 , 3 , (const char *[3]) { "OIL" , "GAS" , "WATER"});
  config_item_set_indexed_selection_set( item , 2 , 2 , (const char *[2]) { "PRODUCER" , "INJECTOR"});



  item = config_add_item( config , "WELL_RATE" , false , true );  /* Group name as part of parsing */
  config_item_set_argc_minmax(item , 4 , 4 , (const config_item_types[4])  {  CONFIG_STRING,         /* GROUP NAME */   
                                                                              CONFIG_STRING ,        /* Well name */
                                                                              CONFIG_FLOAT  ,        /* Corr_length (days) */
                                                                              CONFIG_EXISTING_FILE});/* File with mean , std shift */
}



void load_groups( const config_type * config , const sched_file_type * sched_file , hash_type * group_rates , const sched_history_type * sched_history , const time_t_vector_type * time_vector ) {
  int i;
  for (i=0; i < config_get_occurences( config , "GROUP_RATE" ); i++) {
    const char * group_name   = config_iget( config , "GROUP_RATE" , i , 0 );
    const char * phase_string = config_iget( config , "GROUP_RATE" , i , 1 );
    const char * type_string  = config_iget( config , "GROUP_RATE" , i , 2 );
    const char * min_max_file = config_iget( config , "GROUP_RATE" , i , 3 );
    
    group_rate_type * group_rate = group_rate_alloc( sched_history , time_vector , group_name , phase_string , type_string , min_max_file );
    hash_insert_hash_owned_ref( group_rates , group_name , group_rate , group_rate_free__);
  }


  
  for (i=0; i < config_get_occurences( config , "WELL_RATE" ); i++) {
    const char * group_name   = config_iget( config , "WELL_RATE" , i , 0 );
    const char * well_name    = config_iget( config , "WELL_RATE" , i , 1 );
    double corr_length        = config_iget_as_double( config , "WELL_RATE" , i , 2 );
    const char * stat_file    = config_iget( config , "WELL_RATE" , i , 3 );
    
    well_rate_type * well_rate;
    group_rate_type * group_rate = hash_get( group_rates , group_name );
    well_rate = well_rate_alloc( sched_history , time_vector , well_name , corr_length ,  stat_file , group_rate_get_phase( group_rate) , group_rate_is_producer( group_rate ));
    group_rate_add_well_rate( group_rate , well_rate );
  }
}




void sample( hash_type * group_rates ) {
  hash_iter_type * group_iter = hash_iter_alloc( group_rates );
  
  while (!hash_iter_is_complete( group_iter )) {
    group_rate_type * group_rate = hash_iter_get_next_value( group_iter );
    group_rate_sample( group_rate );
  }

  hash_iter_free( group_iter );
}



int main( int argc , char ** argv ) {
  hash_type   * group_rates = hash_alloc();
  config_type * config      = config_alloc();
  char        * config_file;
  {
    char * config_base;
    char * config_ext;
    char * run_path;
    util_alloc_file_components( argv[1] , &run_path , &config_base , &config_ext);
    if (run_path != NULL) {
      printf("Changing to directory: %s \n",run_path);
      if (chdir( run_path) != 0)
        util_exit("Hmmmm - failed to change to directory:%s \n",run_path);
    }
    config_file = util_alloc_filename(NULL , config_base , config_ext);
    util_safe_free( config_base );
    util_safe_free( config_ext );
    util_safe_free( run_path );
  }
  
  config_init( config );
  config_parse(config , config_file , "--" , NULL , "DEFINE" , false , true );
  {
    const char * data_file       = config_iget( config , "DATA_FILE" , 0 , 0 );
    const char * sched_file_name = config_iget( config , "SCHEDULE_FILE" , 0 , 0 );
    path_fmt_type * sched_fmt    = path_fmt_alloc_path_fmt( config_iget( config , "TARGET" , 0 , 0) );
    const int num_realizations   = config_iget_as_int(config , "NUM_REALIZATIONS" , 0 , 0 );
    msg_type * msg               = msg_alloc("Creating file: ");
    
    time_t start_date = ecl_util_get_start_date( data_file );
    time_t_vector_type * time_vector;
    {
      sched_file_type * sched_file       = sched_file_parse_alloc( sched_file_name , start_date );
      sched_history_type * sched_history = sched_history_alloc(":");
      sched_history_update( sched_history , sched_file );
      
      time_vector = sched_file_alloc_time_t_vector( sched_file );
      load_groups( config , sched_file ,group_rates , sched_history , time_vector );
      sched_history_free( sched_history );
      sched_file_free( sched_file );
    }
    
    {
      
      int i;
      msg_show( msg );
      for (i = 0; i < num_realizations; i++) {
        //sched_file_type * sched_file = sched_file_alloc_copy( );
        sched_file_type * sched_file = sched_file_parse_alloc( sched_file_name , start_date );
        sample( group_rates );
        sched_file_update( sched_file , WCONHIST , perturb_wconhist , group_rates );
        sched_file_update( sched_file , WCONINJE , perturb_wconinje , group_rates );

        {
          char * new_file = path_fmt_alloc_file(sched_fmt , true , i );
          sched_file_fprintf( sched_file , new_file , false);
          msg_update( msg , new_file );
          free( new_file );
        }
        sched_file_free( sched_file );
      }
    }
    msg_free( msg , true );
  }
  config_free( config );
  hash_free( group_rates );
}
