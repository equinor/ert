/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'container_config.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <vector.h>

#include <container_config.h>


struct container_config_struct {
  vector_type * nodes;
};


container_config_type * container_config_alloc( const char * key ) {
  container_config_type * container = util_malloc( sizeof * container , __func__ );
  container->nodes = vector_alloc_new();
  return container;
}



void container_config_free( container_config_type * container_config ) {
  vector_free( container_config->nodes );
  free( container_config );
}



