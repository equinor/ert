/*
   Copyright (C) 2011  Statoil ASA, Norway. 
   
   The file 'well_conn.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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


#ifndef __WELL_CONN_H__
#define __WELL_CONN_H__


#ifdef __cplusplus
extern "C" {
#endif
  
#include <stdbool.h>
#include <ecl_kw.h>
#include <ecl_intehead.h>

  typedef struct  {
    int  i;
    int  j;
    int  k;
    bool open;
  } well_conn_type;
  
  
  void             well_conn_free__( void * arg );
  well_conn_type * well_conn_alloc( const ecl_kw_type * icon_kw , const ecl_intehead_type * header , int well_nr , int conn_nr);
  
#ifdef __cplusplus
}
#endif
#endif
