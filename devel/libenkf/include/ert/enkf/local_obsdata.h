/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'local_obsdata.h'
    
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
#ifndef __LOCAL_OBSDATA_H__
#define __LOCAL_OBSDATA_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <ert/util/type_macros.h>


typedef struct local_obsdata_struct local_obsdata_type;

local_obsdata_type *  local_obsdata_alloc( );
void                  local_obsdata_free( local_obsdata_type * data );


UTIL_IS_INSTANCE_HEADER( local_obsdata );

#ifdef __cplusplus
}
#endif
#endif
