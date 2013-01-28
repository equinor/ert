/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'plot.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

/*char * plot_refcase_fscanf(enkf_sched_type * enkf_sched , FILE * stream , bool * at_eof) {
  char * line = util_fscanf_alloc_line(stream , at_eof);
  return line;
  }*/
#include <ert/util/stringlist.h>

stringlist_type * plot_refcase_fscanf(const char * plot_refcase_file ) {
  stringlist_type * list_of_refcases = stringlist_alloc_new();
  if (plot_refcase_file != NULL){
    FILE * stream = util_fopen(plot_refcase_file , "r");
    bool at_eof;
    do { 
      stringlist_append_copy( list_of_refcases ,util_fscanf_alloc_line(stream , &at_eof));
    } while (!at_eof);
    
    fclose( stream );
  }
  return list_of_refcases;
}
