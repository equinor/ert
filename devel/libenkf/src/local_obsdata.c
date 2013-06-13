/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'local_obsdata.c'
    
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
#include <stdlib.h>

#include <ert/util/util.h>
#include <ert/util/type_macros.h>

#include <ert/enkf/local_obsdata.h>


#define LOCAL_OBSDATA_TYPE_ID 86331309

struct local_obsdata_struct {
  UTIL_TYPE_ID_DECLARATION;
};



UTIL_IS_INSTANCE_FUNCTION( local_obsdata  , LOCAL_OBSDATA_TYPE_ID )

local_obsdata_type * local_obsdata_alloc( ) {
  local_obsdata_type * data = util_malloc( sizeof * data );
  UTIL_TYPE_ID_INIT( data , LOCAL_OBSDATA_TYPE_ID );
  return data;
}



void local_obsdata_free( local_obsdata_type * data ) {
   free( data );
}

