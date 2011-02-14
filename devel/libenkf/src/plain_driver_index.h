/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'plain_driver_index.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __PLAIN_DRIVER_INDEX_H__
#define __PLAIN_DRIVER_INDEX_H__
#include <stdio.h>

typedef struct plain_driver_index_struct plain_driver_index_type;

void * plain_driver_index_fread_alloc(const char *  , FILE * );
void   plain_driver_index_fwrite_mount_info(FILE *  , const char * );

#endif
