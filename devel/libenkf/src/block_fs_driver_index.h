/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'block_fs_driver_index.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __BLOCK_FS_DRIVER_INDEX__
#define __BLOCK_FS_DRIVER_INDEX__


#include <stdio.h>
#include <fs_types.h>
#include <stdbool.h>

typedef struct block_fs_driver_index_struct block_fs_driver_index_type;


void                         block_fs_driver_index_fwrite_mount_info(FILE * stream );
block_fs_driver_index_type * block_fs_driver_index_fread_alloc(const char * root_path , FILE * stream);






#endif
