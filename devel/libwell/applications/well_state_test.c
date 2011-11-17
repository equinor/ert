/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'well_state_test.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/

#include <time.h>
#include <stdbool.h>

#include <util.h>
#include <int_vector.h>
#include <ecl_intehead.h>
#include <ecl_file.h>
#include <ecl_kw.h>
#include <ecl_kw_magic.h>
#include <ecl_util.h>

#include <well_state.h>



int main( int argc , char ** argv) {
  const char * restart_file  = argv[1];

  ecl_file_type * ecl_file   = ecl_file_open( restart_file );
  ecl_intehead_type * header = ecl_intehead_alloc( ecl_file_iget_named_kw( ecl_file , INTEHEAD_KW , 0 ));

  {
    well_state_type * well_state = well_state_alloc( ecl_file , header , 0 , 0 , 0 );
    well_state_free( well_state );
  }
  {
    well_state_type * well_state = well_state_alloc( ecl_file , header , 0 , 0 , 1 );
    well_state_free( well_state );
  }
  {
    well_state_type * well_state = well_state_alloc( ecl_file , header , 0 , 0 , 2 );
    well_state_free( well_state );
  }
 
  ecl_intehead_free( header );
  ecl_file_close( ecl_file );
}
