/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'well_info_test.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <well_info.h>
#include <well_conn.h>

int main( int argc , char ** argv) {
  well_info_type * well_info = well_info_alloc( NULL );
  for (int i=1; i < argc; i++) {
    printf("Loading file: %s \n",argv[i]);
    well_info_load_rstfile( well_info , argv[i]);
  }
  {
    well_state_type * well_state = well_info_get_state_from_report( well_info , "PG-P2" , 350 );
    well_conn_type * conn = well_state_iget_connection( well_state , 0 );
    printf("well_state: %p \n",well_state );
    printf("ijk : %d , %d , %d \n",conn->i , conn->j , conn->k);
    printf("Open: %d  Type:%d \n",well_state_is_open( well_state ), well_state_get_type( well_state ));
  }
  well_info_free( well_info );
}
