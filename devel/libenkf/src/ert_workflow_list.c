/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'ert_workflow_list.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include <hash.h>
#include <stringlist.h>
#include <util.h>

#include <workflow.h>
#include <workflow_job.h>
#include <workflow_joblist.h>

#include <ert_workflow_list.h>

struct ert_workflow_list_struct {
  stringlist_type       * path_list;
  hash_type             * workflows;
  workflow_joblist_type * joblist;
};



ert_workflow_list_type * ert_workflow_list_alloc() {
  ert_workflow_list_type * workflow_list = util_malloc( sizeof * workflow_list );
  workflow_list->path_list = stringlist_alloc_new();
  workflow_list->workflows = hash_alloc();
  workflow_list->joblist   = workflow_joblist_alloc();
  return workflow_list;
}



void ert_workflow_list_free( ert_workflow_list_type * workflow_list ) {
  hash_free( workflow_list->workflows );
  stringlist_free( workflow_list->path_list );
  workflow_joblist_free( workflow_list->joblist );
  free( workflow_list );
}



