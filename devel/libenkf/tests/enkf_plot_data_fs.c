/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'enkf_plot_data_fs.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <stdio.h>
#include <stdbool.h>

#include <ert/util/test_util.h>
#include <ert/util/test_work_area.h>
#include <ert/enkf/enkf_fs.h>
#include <ert/enkf/enkf_main.h>


void test_load( enkf_main_type * enkf_main , const char * casename ) {
  enkf_plot_data_type * plot_data = enkf_plot_data_alloc();
  enkf_plot_data_free( plot_data );
}





int main(int argc, char ** argv) {
  const char * config_file = argv[1];
  test_work_area_type * work_area = test_work_area_alloc( "enkf_main_fs" );
  char * model_config;
  util_alloc_file_components( config_file , NULL , &model_config , NULL);
  test_work_area_copy_parent_content( work_area , config_file );
  {
    const char * site_config = "/project/res/etc/ERT/site-config";
    enkf_main_type * enkf_main = enkf_main_bootstrap( site_config , model_config , false , false );

    test_load(enkf_main , "enkf");
    enkf_main_free( enkf_main );
  }  

  test_work_area_free( work_area );
  exit(0);
}
