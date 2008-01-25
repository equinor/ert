#include <stdlib.h>
#include <util.h>
#include <enkf_fs.h>
#include <path_fmt.h>
#include <enkf_node.h>
#include <basic_driver.h>



struct enkf_fs_struct {
  basic_driver_type  * dynamic_analyzed;
  basic_driver_type  * dynamic_forecast;
  basic_driver_type  * eclipse_static;
  basic_driver_type  * parameter;
};




enkf_fs_type * enkf_fs_alloc(void * dynamic_analyzed , void * dynamic_forecast , void * eclipse_static , void * parameter) {
  enkf_fs_type * fs = malloc(sizeof * fs);

  fs->dynamic_analyzed  = (basic_driver_type *) dynamic_analyzed;   
  fs->dynamic_forecast  = (basic_driver_type *) dynamic_forecast;
  fs->eclipse_static    = (basic_driver_type *) eclipse_static;
  fs->parameter         = (basic_driver_type *) parameter;
  
  basic_driver_assert_cast(fs->dynamic_analyzed);
  basic_driver_assert_cast(fs->dynamic_forecast);
  basic_driver_assert_cast(fs->eclipse_static);
  basic_driver_assert_cast(fs->parameter);
  
  return fs;
}




static basic_driver_type * enkf_fs_select_driver(enkf_fs_type * fs , const enkf_node_type * enkf_node , bool analyzed) {
  enkf_var_type var_type = enkf_node_get_var_type(enkf_node);
  basic_driver_type * driver;
  switch (var_type) {
  case(constant):
    driver = fs->parameter;
    break;
  case(static_parameter):
    driver = fs->parameter;
    break;
  case(parameter):
    driver = fs->parameter;
    break;
  case(ecl_restart):
    if (analyzed)
      driver = fs->dynamic_analyzed;
    else
      driver = fs->dynamic_forecast;
    break;
  case(ecl_summary):
    if (analyzed)
      driver = fs->dynamic_analyzed;
    else
      driver = fs->dynamic_forecast;
    break;
  case(ecl_static):
    driver = fs->eclipse_static;
    break;
  default:
    fprintf(stderr,"%s: fatal internal error - could not determine enkf_fs driver for object - aborting:\n",__func__);
    enkf_node_printf(enkf_node);
    abort();
  }
  basic_driver_assert_cast(driver);
  return driver;
}


static void enkf_fs_free_driver(basic_driver_type * driver) {
  driver->free_driver(driver);
}


void enkf_fs_free(enkf_fs_type * fs) {
  enkf_fs_free_driver(fs->dynamic_analyzed);
  enkf_fs_free_driver(fs->dynamic_forecast);
  enkf_fs_free_driver(fs->parameter);
  enkf_fs_free_driver(fs->eclipse_static);
}


void enkf_fs_swapin_node(enkf_fs_type * enkf_fs , enkf_node_type * enkf_node , int report_step , int iens , bool analyzed) {
  basic_driver_type * driver = enkf_fs_select_driver(enkf_fs , enkf_node , analyzed);
  driver->swapin(driver , report_step , iens , analyzed , enkf_node);
}


void enkf_fs_swapout_node(enkf_fs_type * enkf_fs , enkf_node_type * enkf_node , int report_step , int iens , bool analyzed) {
  basic_driver_type * driver = enkf_fs_select_driver(enkf_fs , enkf_node , analyzed);
  driver->swapout(driver , report_step , iens , analyzed , enkf_node); 
}

