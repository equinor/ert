#include <stdlib.h>
#include <string.h>
#include <sched_types.h>
#include <well_history.h>
#include <double_vector.h>
#include <int_vector.h>
#include <time_t_vector.h>
#include <size_t_vector.h>
#include <bool_vector.h>
#include <sched_kw.h>
#include <sched_kw_wconhist.h>
#include <sched_kw_wconinje.h>
#include <sched_kw_wconinjh.h>


#define WELL_HISTORY_TYPE_ID   44431872


 /**
    The following paradigm applies to the schedule files:

      o It is based on keywords which will apply until the next
        occurence of the same keyword.

      o It is naturaly organized in blocks delimited by DATES / TSTEP
        keywords.


       Example
       -------

       DATES 
          1  'JAN'  2000 /

       WCONHIST 
           'OP_1'      'OPEN'      'ORAT'   7996.000      4.000 1.46402E+006  5* / 
       /

       DATES
          1  'FEB'  2000 /

                                    <--- What is the oil rate here?

       DATES
          1  'MAR'  2000 /


       END


       In this example an historical oil production rate of 7996 is
       specified for immediately following the DATES keyword at
       1. 'JAN' - but what is the oil production rate at 15.th of
       February? The point is that the 7996 rate applies until it is
       set again, i.e. the DATES / TSTEP are not that suitable as block
       units.


    To support this behaviour the vector types from vector_template.c
    are well suited; setting the default with
    e.g. double_vector_set_default() will ascertain that the default
    value is returned for all subsequent gets from the vector beyond
    the set length. (That was clear ....)

*/



struct well_history_struct {
  UTIL_TYPE_ID_DECLARATION;
  char                * well_name;
  int_vector_type     * kw_type;          /* This enum should be one of: NONE(default), WCONHIST , WCONINJE and WCONINJH (sched_kw_type_enum in sched_types.h). */
  size_t_vector_type    * active_state;    /* Contains pointer to the currently active of the xxx_state objects. The size_t_vector instance is abused to store pointer values (i.e. addresses). */
  wconhist_state_type * wconhist_state;         
  wconinje_state_type * wconinje_state;       
  wconinjh_state_type * wconinjh_state;
};



       UTIL_SAFE_CAST_FUNCTION( well_history , WELL_HISTORY_TYPE_ID )
static UTIL_SAFE_CAST_FUNCTION_CONST( well_history , WELL_HISTORY_TYPE_ID )


well_history_type * well_history_alloc( const char * well_name , const time_t_vector_type * time) {
   well_history_type * well_history = util_malloc( sizeof * well_history , __func__);
   UTIL_TYPE_ID_INIT( well_history , WELL_HISTORY_TYPE_ID );
   well_history->well_name       = util_alloc_string_copy( well_name );
   well_history->kw_type         = int_vector_alloc(0 , NONE);
   well_history->wconhist_state  = wconhist_state_alloc( time );
   well_history->wconinje_state  = wconinje_state_alloc( time );
   well_history->wconinjh_state  = wconinjh_state_alloc( time );
   well_history->active_state    = size_t_vector_alloc(0 , 0);
   return well_history;
 }




void well_history_free( well_history_type * well_history ) {
   free( well_history->well_name );
   int_vector_free( well_history->kw_type );
   wconhist_state_free( well_history->wconhist_state );
   wconinjh_state_free( well_history->wconinjh_state );
   wconinje_state_free( well_history->wconinje_state );
   size_t_vector_free( well_history->active_state );
   free( well_history );
}


void well_history_free__( void * arg ) {
  well_history_free( well_history_safe_cast( arg ));
}



 /*****************************************************************/



 void well_history_add_keyword( well_history_type * well_history, const sched_kw_type * sched_kw , int  report_step ) {
   sched_kw_type_enum new_type     = sched_kw_get_type( sched_kw );
   sched_kw_type_enum current_type = int_vector_safe_iget( well_history->kw_type , report_step );
   
   if ((new_type != current_type) && (current_type != NONE)) {
     /* 
        The well is changing type and we must "close" the current
        status first.
     */
     switch( current_type ) {
     case( WCONHIST ):
       sched_kw_wconhist_close_state( well_history->wconhist_state , report_step );
       break;
     case( WCONINJH):
       sched_kw_wconinjh_close_state( well_history->wconinjh_state , report_step );
       break;
     default:
       break;
     }
   }
   
   switch( new_type ) {
   case(WCONHIST):
     int_vector_iset_default( well_history->kw_type , report_step , WCONHIST );
     size_t_vector_iset_default( well_history->active_state , report_step , ( long ) well_history->wconhist_state );
     sched_kw_wconhist_update_state(sched_kw_get_const_data( sched_kw ) , well_history->wconhist_state , well_history->well_name , report_step );
     break;
   case(WCONINJH):
     int_vector_iset_default( well_history->kw_type , report_step , WCONINJH );
     size_t_vector_iset_default( well_history->active_state , report_step , ( long ) well_history->wconinjh_state );
     sched_kw_wconinjh_update_state(sched_kw_get_const_data( sched_kw ) , well_history->wconinjh_state , well_history->well_name , report_step );
     break;
   default:
     break;
     // Nothing wrong with this (I think) ??
   }
 }

/*****************************************************************/

wconhist_state_type * well_history_get_wconhist( well_history_type * well_history ) {
  return well_history->wconhist_state;
}

/*****************************************************************/


sched_kw_type_enum well_history_iget_active_kw( const well_history_type * well_history , int report_step ) {
  return int_vector_safe_iget( well_history->kw_type , report_step );
}


const void * well_history_get_state_ptr( const well_history_type * well_history , sched_kw_type_enum kw_type ) {
  switch( kw_type ) {
  case(WCONHIST):
    return well_history->wconhist_state;
    break;
  case(WCONINJH):
    return well_history->wconinjh_state;
    break;
  case(WCONINJE):
    return well_history->wconinje_state;
    break;
  default:
    util_abort("%s: non-handled enum value \n",__func__);
    return NULL;
  }
}






double well_history_iget( well_index_type * index , int report_step ) {
  const well_history_type * well_history  = well_history_safe_cast_const( well_index_get_state( index ));
  sched_kw_type_enum current_type         = int_vector_safe_iget( well_history->kw_type , report_step );
  sched_history_callback_ftype * func     = well_index_get_callback( index , current_type );
  
  if (func != NULL) {
    void * state_ptr = (void *) size_t_vector_safe_iget( well_history->active_state , report_step );
    return func( state_ptr , report_step );
  } else
    return -1;
}
