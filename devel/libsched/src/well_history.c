#include <stdlib.h>
#include <string.h>
#include <sched_types.h>
#include <well_history.h>
#include <double_vector.h>
#include <int_vector.h>
#include <time_t_vector.h>
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
   int_vector_type     * kw_type;          /* This enum should be one of: NONE(default), WCONHIST , WCONINJE and WCONINJH. */
   wconhist_state_type * wconhist_state;         
   wconinje_state_type * wconinje_state;       
   wconinjh_state_type * wconinjh_state;
 };



 UTIL_SAFE_CAST_FUNCTION( well_history , WELL_HISTORY_TYPE_ID )


well_history_type * well_history_alloc( const char * well_name ) {
   well_history_type * well_history = util_malloc( sizeof * well_history , __func__);
   UTIL_TYPE_ID_INIT( well_history , WELL_HISTORY_TYPE_ID );
   well_history->well_name = util_alloc_string_copy( well_name );

   well_history->wconhist_state  = wconhist_state_alloc( );
   well_history->wconinje_state  = wconinje_state_alloc( );
   well_history->wconinjh_state  = wconinjh_state_alloc( );

   return well_history;
 }




void well_history_free( well_history_type * well_history ) {
   free( well_history->well_name );
   int_vector_free( well_history->kw_type );
   wconhist_state_free( well_history->wconhist_state );
   wconinjh_state_free( well_history->wconinjh_state );
   wconinje_state_free( well_history->wconinje_state );
   free( well_history );
 }


void well_history_free__( void * arg ) {
  well_history_free( well_history_safe_cast( arg ));
}


 /*****************************************************************/

 void well_history_add_keyword( well_history_type * well_history, const sched_kw_type * sched_kw , const char * well_name , int  report_step ) {
   sched_kw_type_enum kw_type = sched_kw_get_type( sched_kw );
   switch( kw_type ) {
   case(WCONHIST):
     int_vector_iset_default( well_history->kw_type , report_step , WCONHIST );
     sched_kw_wconhist_update_state(sched_kw_get_const_data( sched_kw ) , well_history->wconhist_state , well_name , report_step );
     break;
   case(WCONINJH):
     int_vector_iset_default( well_history->kw_type , report_step , WCONINJH );
     sched_kw_wconinjh_update_state(sched_kw_get_const_data( sched_kw ) , well_history->wconinjh_state , well_name , report_step );
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
