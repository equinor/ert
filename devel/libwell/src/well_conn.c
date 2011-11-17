/*
   Copyright (C) 2011  Statoil ASA, Norway. 
   
   The file 'well_conn.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <ecl_kw.h>
#include <stdbool.h>
#include <well_const.h>
#include <well_conn.h>


well_conn_type * well_conn_alloc( const ecl_kw_type * icon_kw , const ecl_intehead_type * header , int well_nr , int conn_nr) {
  const int icon_offset = header->niconz * ( header->ncwmax * well_nr + conn_nr );
  well_conn_type * conn = util_malloc( sizeof * conn , __func__ );
  
  conn->i = ecl_kw_iget_int( icon_kw , icon_offset + ICON_I_ITEM );
  conn->j = ecl_kw_iget_int( icon_kw , icon_offset + ICON_J_ITEM );
  conn->k = ecl_kw_iget_int( icon_kw , icon_offset + ICON_K_ITEM );
  {
    int int_status = ecl_kw_iget_int( icon_kw , icon_offset + ICON_STATUS_ITEM );
    if (int_status > 0)
      conn->open = true;
    else
      conn->open = false;
  }

  return conn;
}


void well_conn_free( well_conn_type * conn) {
  free( conn );
}


void well_conn_free__( void * arg ) {
  well_conn_type * conn = (well_conn_type *) arg;
  well_conn_free( conn );
}
