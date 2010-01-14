#include <stdlib.h>
#include <string.h>
#include <sched_types.h>
#include <well_state.h>
#include <double_vector.h>
#include <int_vector.h>
#include <time_t_vector.h>
#include <bool_vector.h>
#include <sched_kw_wconhist.h>

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


/*
  The well_state implemented in this file is built after a normal
  parsing of the schedule file.
*/



/**
   This struct contains the information learned from parsing the
   WCONHIST keyword.
*/

typedef struct  {
  int_vector_type    * state;                   /* Contains values from the well_status_enum. */ 
  int_vector_type    * cmode;                   /* Contains values from the well_cm_enum. */
  double_vector_type * oil_rate;                
  double_vector_type * gas_rate;                
  double_vector_type * water_rate;              
  double_vector_type * thp;
  double_vector_type * bhp;
  double_vector_type * wgas_rate;
} producer_state_type;


/*
  Parsing the WCONINJE keyword. 
*/
typedef struct {
  int_vector_type    * phase;                  /* Contains values from sched_phase_enum */
  int_vector_type    * state;                  /* Contains values from the well_status_enum. */ 
  int_vector_type    * cmode;                  /* Contains values from the well_cm_enum. */
  double_vector_type * surface_flow;
  double_vector_type * reservoir_flow;
  double_vector_type * bhp_limit;
  double_vector_type * thp_limit;
} injector_state_type;




struct well_state_struct {
  int_vector_type     * well_type;              /* Injector or producer - using enum well_state_well_type from sched_types. */
  producer_state_type * producer_state;         
  injector_state_type * injector_state;          
  
};


/*****************************************************************/
/*
  Functions implementing the producer state.
*/

static producer_state_type * producer_state_alloc( ) {
  producer_state_type * producer = util_malloc( sizeof * producer , __func__);
  producer->state      = int_vector_alloc( 0 , WCONHIST_DEFAULT_STATUS );
  producer->cmode      = int_vector_alloc( 0 , 0 ); 
  
  producer->oil_rate   = double_vector_alloc( 0 , 0 );
  producer->gas_rate   = double_vector_alloc( 0 , 0 );
  producer->water_rate = double_vector_alloc( 0 , 0 );
  producer->thp        = double_vector_alloc( 0 , 0 );
  producer->bhp        = double_vector_alloc( 0 , 0 );
  producer->wgas_rate  = double_vector_alloc( 0 , 0 );
  
  return producer;
}


static void producer_state_free( producer_state_type * producer ) {
  int_vector_free( producer->state );
  int_vector_free( producer->cmode );
  
  double_vector_free( producer->oil_rate );
  double_vector_free( producer->gas_rate );
  double_vector_free( producer->water_rate );
  double_vector_free( producer->thp );
  double_vector_free( producer->bhp );
  double_vector_free( producer->wgas_rate );
  
  free( producer );
}


/*****************************************************************/
/*
  Functions implementing the injector state.
*/
