#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <util.h>
#include <time_t_vector.h>
#include <double_vector.h>
#include <stdbool.h>
#include <time.h>
#include <pert_util.h>

static inline int randint() {
  return rand();
}


void rand_dbl(int N , double max , double *R) {
  int i;
  for (i=0; i < N; i++) 
    R[i] = randint() * max / RAND_MAX;
}


double rand_normal(double mean , double std) {
  const double pi = 3.141592653589;
  double R[2];
  rand_dbl(2 , 1.0 , R);
  return mean + std * sqrt(-2.0 * log(R[0])) * cos(2.0 * pi * R[1]);
}


void rand_stdnormal_vector(int size , double *R) {
  int i;
  for (i = 0; i < size; i++)
    R[i] = rand_normal(0.0 , 1.0);
}





/*****************************************************************/

static void set_ts(const time_t_vector_type * time_vector , double_vector_type * values , time_t start_date , time_t end_date , double value, bool_vector_type * tsp , bool percent) {
  int i;
  for (i=0; i < time_t_vector_size( time_vector ); i++) {
    time_t t = time_t_vector_iget( time_vector , i );
    if ((t >= start_date) && (t < end_date)) {
      double_vector_iset( values , i , value );
      if (tsp != NULL) {
        bool_vector_iset( tsp , i , percent );
      }
    }
  }
}

/**
   File format:

   *           -  12/07/2009   500  577 
   12/07/2009  -  16/09/2009   672  666
   17/09/2009  -      *        100   10%   
   

   1. Both dates can be replaced with '*' - which is implied to mean
      either the start date, or the end date.

   2. The formatting of the data strings is 100% NAZI - no spaces
      allowed.
     
   3. The date intervals are half-open, [date1,date2).

   4. The date lines can overlap - they are applied in line-order.

   5. The last float value (i.e. 577 and 666 on the liness above) can
      be a percent value; that should indicated with '%' immediately
      following the number. The percent number should be in the
      interval [0,100]. The bool vector tsp is updated to indicate
      whether the data should be interpreted as percent or not.
      
*/


static void load_exit( FILE * stream , const char * filename) {

  fprintf(stderr," Something wrong around line:%d of file:%s\n",util_get_current_linenr( stream ) , filename );
  fprintf(stderr," Each line should be:\n  date1 - date2  value\nwhere date should be formatted as 12/06/2003. \n");
  exit(1);

}

   
void fscanf_2ts(const time_t_vector_type * time_vector , const char * filename , double_vector_type * ts1 , double_vector_type * ts2, bool_vector_type * tsp) {
  time_t start_time       = time_t_vector_get_first( time_vector );
  time_t end_time         = time_t_vector_get_last( time_vector ) + 1;

  {
    FILE * stream = util_fopen( filename , "r");
    char datestring1[32];
    char datestring2[32];
    char dash;
    double value1 , value2;
    char value2string[32];
      
    while (true) {
      int read_count = fscanf(stream , "%s %c %s %lg %s" , datestring1 , &dash , datestring2, &value1 , value2string);
      if (read_count == 5) {
        bool   OK = true;
        bool   percent = false;
        time_t t1      = -1;
        time_t t2      = -1;
        if (util_string_equal( datestring1 , "*"))
          t1 = start_time;
        else
          OK = util_sscanf_date( datestring1 , &t1 );
        
        if (util_string_equal( datestring2 , "*"))
          t2 = end_time;
        else
          OK = (OK && util_sscanf_date( datestring2 , &t2 ));

        {
          char * error_ptr;
          value2 = strtod( value2string , &error_ptr);
          if (error_ptr[0] == '%') 
            percent = true;
          else {
            if (error_ptr[0] != '\0')
              OK = false;
          }
        }
        
        if (OK) {
          set_ts( time_vector , ts1 , t1 , t2 , value1  , NULL , false  );
          set_ts( time_vector , ts2 , t1 , t2 , value2  , tsp  , percent);
        } else 
          load_exit( stream  , filename ); 

      } else {
        if (read_count == EOF)
          break;
        else
          load_exit( stream , filename );
      }
    } 
    fclose( stream );
  }
}



/*****************************************************************/



