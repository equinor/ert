/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'ert_report_list.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <util.h>
#include <stringlist.h>
#include <vector.h>
#include <subst_list.h>

#include <ert_report.h>
#include <ert_report_list.h>


struct ert_report_list_struct {
  stringlist_type * path_list;
  vector_type     * report_list;
  char            * target_path;
};



ert_report_list_type * ert_report_list_alloc(const char * target_path) {
  ert_report_list_type * report_list = util_malloc( sizeof * report_list , __func__);
  report_list->path_list   = stringlist_alloc_new( );
  report_list->report_list = vector_alloc_new( );
  report_list->target_path = NULL;
  ert_report_list_set_target_path( report_list , target_path );
  return report_list;
}


void ert_report_list_set_target_path( ert_report_list_type * report_list , const char * target_path ) {
  report_list->target_path = util_realloc_string_copy( report_list->target_path , target_path );
}


bool ert_report_list_add_path( ert_report_list_type * report_list , const char * path ) {
  if (util_is_directory( path )) {
    stringlist_append_copy( report_list->path_list , path );
    return true;
  } else {
    fprintf(stderr,"** Warning: Path:%s does not exist - not added to report template search path.\n",path);
    return false;
  }
}


static void ert_report_list_add__( ert_report_list_type * report_list , const char * template_path , const char * target_name) {
  ert_report_type * report = ert_report_alloc(template_path , target_name);

  vector_append_owned_ref( report_list->report_list , report , ert_report_free__ );
}


bool ert_report_list_add_report( ert_report_list_type * report_list , const char * template_path) {
  // Split on ":" first ...
  char * target_name = NULL;

  if (util_is_file( template_path )) {
    ert_report_list_add__( report_list , template_path , target_name);
    return true;
  } else {
    for (int i=0; i < stringlist_get_size( report_list->path_list ); i++) {
      const char * template_file = util_alloc_filename( stringlist_iget( report_list->path_list , i ) , template_path , NULL);
      if (util_is_file( template_file )) {
        ert_report_list_add__( report_list , stringlist_iget( report_list->path_list , i ) , target_name );
        return true;  // Jump out of the whole thing ...
      }
    }
    return false;
  }
}


void ert_report_list_free( ert_report_list_type * report_list ){ 
  stringlist_free( report_list->path_list );
  vector_free( report_list->report_list );
  free( report_list );
}

