/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'plain_driver_common.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __PLAIN_DRIVER_COMMON_H__
#define __PLAIN_DRIVER_COMMON_H__
#include <path_fmt.h>
#include <enkf_types.h>
#include <enkf_node.h>

void 		plain_driver_common_load_node(const char * ,  int , int , state_enum , enkf_node_type * );
path_fmt_type * plain_driver_common_realloc_path_fmt(path_fmt_type *  , const char *  , const char * , const char * );


#endif
