#include <stdio.h>
#include <string.h>
#include <util.h>
#include <hash.h>
#include <ecl_sum.h>
#include <ecl_util.h>
#include <stringlist.h>
#include <bool_vector.h>
#include <gruptree.h>
#include <sched_history.h>
#include <history.h>


struct history_struct{
  const ecl_sum_type    * ecl_sum;        /* ecl_sum instance used when the data are taken from a summary instance. Observe that this is NOT owned by history instance.*/
  sched_history_type    * sched_history;
  history_source_type     source;
};


history_source_type history_get_source_type( const char * string_source ) {
  history_source_type source_type;

  if (strcmp( string_source , "REFCASE_SIMULATED") == 0)
    source_type = REFCASE_SIMULATED;
  else if (strcmp( string_source , "REFCASE_HISTORY") == 0)
    source_type = REFCASE_HISTORY;
  else if (strcmp( string_source , "SCHEDULE") == 0)
    source_type = SCHEDULE;
  else
    util_abort("%s: Sorry source:%s not recognized\n",__func__ , string_source);

  return source_type;
}


const char * history_get_source_string( history_source_type history_source ) {
  switch( history_source ) {
  case( REFCASE_SIMULATED ):
    return "REFCASE_SIMULATED";
    break;
  case(REFCASE_HISTORY ):
    return "REFCASE_HISTORY";
    break;
  case(SCHEDULE ):
    return "SCHEDULE";
    break;
  default: 
    util_abort("%s: internal fuck up \n",__func__);
    return NULL;
  }
}






static history_type * history_alloc_empty()
{
  history_type * history = util_malloc(sizeof * history, __func__);
  history->ecl_sum       = NULL; 
  return history;
}



/******************************************************************/
// Exported functions for manipulating history_type. Acess functions further below.


void history_free(history_type * history)
{
  sched_history_free( history->sched_history );
  free(history);
}


history_type * history_alloc_from_sched_file(const char * sep_string , const sched_file_type * sched_file)
{
  history_type * history = history_alloc_empty( );
  history->sched_history = sched_history_alloc( sep_string );
  sched_history_update( history->sched_history , sched_file );
  history->source = SCHEDULE;
  
  return history;
}



void history_use_summary(history_type * history, const ecl_sum_type * refcase , bool use_h_keywords) {

  history->ecl_sum = refcase;     /* This function does not really do anthing - it just sets the ecl_sum field of the history instance. */
  if (use_h_keywords)
    history->source = REFCASE_HISTORY;
  else
    history->source = REFCASE_SIMULATED;
  
}




/******************************************************************/
// Exported functions for accessing history_type.




/**
  Get the number of restart files the underlying schedule file would produce.
*/
int history_get_num_restarts(const history_type * history)
{
  return sched_history_get_last_history( history->sched_history );
}










void history_init_ts( const history_type * history , const char * summary_key , double_vector_type * value, bool_vector_type * valid) {
  double_vector_reset( value );
  bool_vector_reset( valid );
  bool_vector_set_default( valid , false);
  if (history->source == SCHEDULE) {
    for (int tstep = 0; tstep <= sched_history_get_last_history(history->sched_history); tstep++) {
      if (sched_history_open( history->sched_history , summary_key , tstep)) {
        bool_vector_iset( valid , tstep , true );
        double_vector_iset( value , tstep , sched_history_iget( history->sched_history , summary_key , tstep));
      } else
        bool_vector_iset( valid , tstep , false );
    }
  } else {
    char * local_key;
    if (history->source == REFCASE_HISTORY) {
      /* Must create a new key with 'H' for historical values. */
      const ecl_smspec_type * smspec      = ecl_sum_get_smspec( history->ecl_sum );
      const char            * join_string = ecl_smspec_get_join_string( smspec ); 
        
      local_key = util_alloc_sprintf( "%sH%s%s" , ecl_sum_get_keyword( history->ecl_sum , summary_key ) , join_string , ecl_sum_get_wgname( history->ecl_sum , summary_key ));
    } else
      local_key = (char *) summary_key;
  
    {
      for (int tstep = 0; tstep <= sched_history_get_last_history(history->sched_history); tstep++) {
        int ministep   = ecl_sum_get_report_ministep_end( history->ecl_sum , tstep );
        if (ministep >= 0) {
          double_vector_iset( value , tstep , ecl_sum_get_general_var( history->ecl_sum , ministep , local_key ));
          bool_vector_iset( valid , tstep , true );
        } else
          bool_vector_iset( valid , tstep , false );    /* Did not have this report step */
      }
    }
    
    if (history->source == REFCASE_HISTORY) 
      free( local_key );
  }
}




/* Uncertain about the first node - offset problems +++ ?? 
   Changed to use node_end_time() at svn ~ 2850

   Changed to sched_history at svn ~2940
*/
time_t history_get_time_t_from_restart_nr( const history_type * history , int restart_nr) {
  return sched_history_iget_time_t( history->sched_history , restart_nr);
}




