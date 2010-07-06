#include <stdlib.h>
#include <util.h>
#include <time_t_vector.h>
#include <double_vector.h>
#include <bool_vector.h>
#include <time.h>
#include <math.h>
#include <pert_util.h>
#include <well_rate.h>
#include <sched_types.h>
#include <sched_kw_wconinje.h>

#define WELL_RATE_ID  6681055


struct well_rate_struct {
  UTIL_TYPE_ID_DECLARATION;
  char * name;
  double                     corr_length;
  double_vector_type       * shift;
  double_vector_type       * mean_shift;
  double_vector_type       * std_shift;
  double_vector_type       * rate; 
  bool_vector_type         * well_open;
  bool_vector_type         * percent_std;
  sched_phase_enum           phase;
  const time_t_vector_type * time_vector;
};
  





void well_rate_update_wconhist( well_rate_type * well_rate , sched_kw_wconhist_type * kw, int restart_nr ) {
  double shift = double_vector_iget( well_rate->shift , restart_nr );
  switch (well_rate->phase) {
  case(OIL):
    sched_kw_wconhist_shift_orat( kw , well_rate->name , shift);
    break;                                               
  case(GAS):                                             
    sched_kw_wconhist_shift_grat( kw , well_rate->name , shift);
    break;                                               
  case(WATER):                                           
    sched_kw_wconhist_shift_wrat( kw , well_rate->name , shift);
    break;
  }
}


void well_rate_update_wconinje( well_rate_type * well_rate , sched_kw_wconinje_type * kw, int restart_nr ) {
  sched_kw_wconinje_shift_surface_flow( kw , well_rate->name , double_vector_iget( well_rate->shift , restart_nr ));
  return;
}




/*
        a = exp(-(t_i - t_(i-1)) / corr_length)
     y(i) = a*y(i - 1) + (1 - a) * N(mean(i) , std(i))
     
*/

void well_rate_sample_shift( well_rate_type * well_rate ) {
  int size   = time_t_vector_size( well_rate->time_vector );
  double * R = util_malloc( size * sizeof * R , __func__);
  int i;
  rand_stdnormal_vector( size , R );
  for (i=0; i < size; i++) 
    R[i] = R[i] * double_vector_iget( well_rate->std_shift , i ) + double_vector_iget( well_rate->mean_shift , i );
  
  double_vector_iset( well_rate->shift , 0 , R[0]);
  
  for (i=1; i < size; i++) {
    double dt    = 1.0 * (time_t_vector_iget( well_rate->time_vector , i ) - time_t_vector_iget( well_rate->time_vector , i - 1)) / (24 * 3600);  /* Days */
    double a     = exp(-dt / well_rate->corr_length );
    double shift = a * double_vector_iget( well_rate->shift , i - 1 ) + (1 - a) * R[i];
    /* The time series is sampled - irrespective of whether the well is open or not. */

    if (bool_vector_iget( well_rate->well_open , i))
      double_vector_iset( well_rate->shift , i , shift );
    else
      double_vector_iset( well_rate->shift , i , 0);
      
  }
  free( R );
}


void well_rate_ishift( well_rate_type * well_rate ,  int index, double shift) {
  if (bool_vector_iget(well_rate->well_open , index))
    double_vector_iadd( well_rate->shift , index , shift );
}


well_rate_type * well_rate_alloc(const time_t_vector_type * time_vector , const sched_file_type * sched_file , const char * name , double corr_length , const char * filename, sched_phase_enum phase, bool producer) {
  well_rate_type * well_rate = util_malloc( sizeof * well_rate , __func__);
  UTIL_TYPE_ID_INIT( well_rate , WELL_RATE_ID );
  well_rate->name         = util_alloc_string_copy( name );
  well_rate->time_vector  = time_vector;
  well_rate->corr_length  = corr_length;
  well_rate->shift        = double_vector_alloc(0,0);
  well_rate->mean_shift   = double_vector_alloc(0 , 0);
  well_rate->std_shift    = double_vector_alloc(0 , 0);
  well_rate->well_open    = bool_vector_alloc(0 , false );
  well_rate->rate         = double_vector_alloc(0 , 0);
  well_rate->phase        = phase;
  well_rate->percent_std  = bool_vector_alloc( 0 , false );
  fscanf_2ts( time_vector , filename , well_rate->mean_shift , well_rate->std_shift , well_rate->percent_std);

  {
    int i;
    for (i=0; i < time_t_vector_size( time_vector ); i++) {
      bool well_open = sched_file_well_open( sched_file , i , well_rate->name);
      bool_vector_iset( well_rate->well_open , i , well_open);
      if (bool_vector_iget( well_rate->percent_std, i)) {
        if (well_open) {
          double rate;
          if (producer) 
            rate = sched_file_well_wconhist_rate( sched_file , i ,well_rate->name);
          else
            rate = sched_file_well_wconinje_rate( sched_file , i ,well_rate->name);
          
          double_vector_imul( well_rate->std_shift , i , rate * 0.01);
        }
      }
    }
  }

  return well_rate;
}


bool well_rate_well_open( const well_rate_type * well_rate , int index ) {
  return bool_vector_iget( well_rate->well_open , index );
}


static UTIL_SAFE_CAST_FUNCTION( well_rate , WELL_RATE_ID );

void well_rate_free( well_rate_type * well_rate ) {
  free( well_rate->name );
  double_vector_free( well_rate->shift );
  double_vector_free( well_rate->mean_shift );
  double_vector_free( well_rate->std_shift );
  bool_vector_free( well_rate->percent_std );
  free( well_rate );
}

void well_rate_free__( void * arg ) {
  well_rate_type * well_rate = well_rate_safe_cast( arg );
  well_rate_free( well_rate );
}



sched_phase_enum well_rate_get_phase( const well_rate_type * well_rate ) {
  return well_rate->phase;
}


const char * well_rate_get_name( const well_rate_type * well_rate ) {
  return well_rate->name;
}


double_vector_type * well_rate_get_shift( well_rate_type * well_rate ) {
  return well_rate->shift;
}
