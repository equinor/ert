/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'ert_report.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <type_macros.h>
#include <util.h>
#include <template.h>
#include <latex.h>
#include <subst_list.h>

#include <ert_report.h>

/*
  The LATEX_PATH_FMT is a format string for where the latex
  compilation should take place. This format string will be passed to
  the mkdtemp() function, where the last 'XXXXXX' characters will be
  replaced with random characters. 

  The mkdtemp() function will only create one - directory, i.e. the
  format string "/tmp/latex/XXXXXX" will fail unless the /tmp/latex
  directory already exists prior to the call.  
*/
#define LATEX_PATH_FMT "/tmp/latex-XXXXXX"  

#define ERT_REPORT_TYPE_ID 919191653

struct ert_report_struct {
  UTIL_TYPE_ID_DECLARATION;
  template_type * template;
  char          * latex_basename;
  char          * input_path;
  char          * debug_path;
  bool            with_xref;
  bool            ignore_errors;
};


static UTIL_SAFE_CAST_FUNCTION( ert_report , ERT_REPORT_TYPE_ID )


ert_report_type * ert_report_alloc( const char * source_file , const char * target_file ) {
  ert_report_type * report = util_malloc( sizeof * report , __func__ );
  report->template = template_alloc( source_file , true , NULL );

  {
    char * input_path;
    if (target_file == NULL) {
      util_alloc_file_components( source_file , &input_path , &report->latex_basename , NULL);
    } else {
      util_alloc_file_components( source_file , &input_path , NULL , NULL);
      util_alloc_file_components( target_file , NULL , &report->latex_basename , NULL);
    }

    if (input_path == NULL)
      report->input_path = util_alloc_cwd();
    else 
      report->input_path = util_alloc_abs_path(input_path);
    
    util_safe_free( input_path );
  }
  report->debug_path = NULL;
  return report;
}





bool ert_report_create( ert_report_type * ert_report , const subst_list_type * context , const char * target_file) {
  char * latex_path = util_alloc_string_copy( LATEX_PATH_FMT );
  bool   success;
  
  if (mkdtemp( latex_path ) == NULL)
    util_abort("%s: failed to create runpath for LaTeX \n",__func__);
  {
    char * latex_file = util_alloc_filename( latex_path , ert_report->latex_basename , LATEX_EXTENSION );
    template_instansiate( ert_report->template , latex_file , context , true );
    {
      latex_type * latex = latex_alloc( latex_file , true );
      latex_link_directory_content( latex , ert_report->input_path );
      success = latex_compile( latex , ert_report->ignore_errors , ert_report->with_xref);

      if (success) 
        util_copy_file( latex_get_target_file( latex ) , target_file );
      else 
        ert_report->debug_path = util_realloc_string_copy( ert_report->debug_path , latex_get_runpath( latex ));
      
      latex_free( latex );
    }
    free( latex_file );
  }
  if (success)
    util_clear_directory( latex_path , true , true );
  
  free( latex_path );
  return success;
}



void ert_report_free( ert_report_type * ert_report ) {

  template_free( ert_report->template );
  free( ert_report->latex_basename );
  free( ert_report->input_path );
  util_safe_free( ert_report->debug_path );

  free( ert_report );
}


void ert_report_free__(void * arg) {
  ert_report_type * ert_report = ert_report_safe_cast( arg );
  ert_report_free( ert_report );
}
