#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <gen_data.h>
#include <gen_data_config.h>
#include <util.h>
#include <ecl_grid.h>
#include <pilot_point_config.h>
#include <pilot_point.h>
#include <forward_model.h>


int main(void) {
  ext_joblist_type * ext_joblist = ext_joblist_alloc();
  ext_joblist_add_job( ext_joblist , "CONVERT" 	    , "/d/proj/bg/enkf/Config/CONVERT");
  ext_joblist_add_job( ext_joblist , "ECHO"    	    , "/d/proj/bg/enkf/Config/ECHO");
  ext_joblist_add_job( ext_joblist , "COPY_FILE"    , "/d/proj/bg/enkf/Config/COPY_FILE");
  
  forward_model_type * f = forward_model_alloc("CONVERT COPY_FILE(SRC_FILE = /tmp/test , TARGET_FILE=/tmp/jalla, HH=ARG) ECHO"  , ext_joblist , true);
  forward_model_python_fprintf(f , "/tmp" , NULL);
  forward_model_free( f );
    ext_joblist_free(ext_joblist);
}



