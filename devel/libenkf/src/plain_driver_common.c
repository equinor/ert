/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'plain_driver_common.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <sys/file.h>
#include <enkf_types.h>
#include <enkf_node.h>
#include <stdio.h>
#include <util.h>
#include <plain_driver_common.h>
#include <path_fmt.h>
#include <buffer.h>

/**
   This file implements small functions which are common to to 
   all the plain_driver_xxxx drivers. 
*/


/**
   See the documentation in the top of plain_driver_dynamic for the
   various parts in the path_fmt instance.

   For compatibility this function will accept NULL for the
   directory. Should be phased out.
*/
path_fmt_type * plain_driver_common_realloc_path_fmt(path_fmt_type * path_fmt , const char * root_path , const char * directory , const char * fmt) {
  char * tmp;
  path_fmt_type * new_path_fmt;
  
  if (path_fmt != NULL) path_fmt_free(path_fmt); /* Free old instance. */

  if (directory != NULL) {
    if (root_path != NULL) 
      tmp = util_alloc_sprintf("%s%c%s%c%s" , root_path , UTIL_PATH_SEP_CHAR , directory , UTIL_PATH_SEP_CHAR , fmt);
    else
      tmp = util_alloc_sprintf("%s%c%s" , directory , UTIL_PATH_SEP_CHAR , fmt); /* Is root_path == NULL at all an option ?? */
  } else {
    if (root_path != NULL) 
      tmp = util_alloc_sprintf("%s%c%s" , root_path , UTIL_PATH_SEP_CHAR , fmt);
    else
      tmp = util_alloc_string_copy( fmt );
  }
  

  new_path_fmt = path_fmt_alloc_directory_fmt( tmp );
  free(tmp);

  return new_path_fmt;
}
