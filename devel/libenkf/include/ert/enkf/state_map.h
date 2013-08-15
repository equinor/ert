/*
   Copyright (C) 2013  Statoil ASA, Norway. 
   The file 'state_map.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __STATE_MAP_H__
#define __STATE_MAP_H__

#ifdef __cplusplus 
extern "C" {
#endif 

#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_types.h>

  typedef struct state_map_struct state_map_type;


  state_map_type         * state_map_alloc( );
  void                     state_map_free( state_map_type * map );
  int                      state_map_get_size( state_map_type * map);
  realisation_state_enum   state_map_iget( state_map_type * map , int index);
  void                     state_map_iset( state_map_type * map ,int index , realisation_state_enum state);


  UTIL_IS_INSTANCE_HEADER( state_map );


#ifdef __cplusplus 
}
#endif
#endif
