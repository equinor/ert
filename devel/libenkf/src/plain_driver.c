#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <plain_driver.h>
#include <plain_driver_parameter.h>
#include <plain_driver_dynamic.h>
#include <plain_driver_static.h>
#include <plain_driver_index.h>


void plain_driver_fwrite_mount_info(FILE * stream) {
  /* The read drivers */
  plain_driver_parameter_fwrite_mount_info(stream , true , "%04d/mem%03d/Parameter"); 
  plain_driver_static_fwrite_mount_info(stream    , true , "%04d/mem%03d/Static"); 
  plain_driver_dynamic_fwrite_mount_info(stream   , true , "%04d/mem%03d/Forecast", "%04d/mem%03d/Analyzed");
  plain_driver_index_fwrite_mount_info(stream     , true , "%04d/mem%03d/INDEX");

  
  /* The write drivers */
  plain_driver_parameter_fwrite_mount_info(stream , false , "%04d/mem%03d/Parameter"); 
  plain_driver_static_fwrite_mount_info(stream    , false , "%04d/mem%03d/Static"); 
  plain_driver_dynamic_fwrite_mount_info(stream   , false , "%04d/mem%03d/Forecast", "%04d/mem%03d/Analyzed");
  plain_driver_index_fwrite_mount_info(stream     , false , "%04d/mem%03d/INDEX");
}
