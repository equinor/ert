#include <stdlib.h>
#include <util.h>
#include <enkf_fs.h>
#include <path_fmt.h>
#include <enkf_node.h>
#include <basic_driver.h>
#include <fs_index.h>



struct enkf_fs_struct {
  basic_driver_type  * dynamic_analyzed;
  basic_driver_type  * dynamic_forecast;
  basic_driver_type  * eclipse_static;
  basic_driver_type  * parameter;
  fs_index_type      * index;
};




enkf_fs_type * enkf_fs_alloc(fs_index_type * fs_index, 
			     void * dynamic_analyzed , void * dynamic_forecast , void * eclipse_static , void * parameter) {
  enkf_fs_type * fs     = util_malloc(sizeof * fs , __func__);
  fs->index             = fs_index;
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




static basic_driver_type * enkf_fs_select_driver(enkf_fs_type * fs , const enkf_config_node_type * config_node , state_enum state) {
  enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
  basic_driver_type * driver = NULL;
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
    if (state == analyzed)
      driver = fs->dynamic_analyzed;
    else if (state == forecast)
      driver = fs->dynamic_forecast;
    else 
      util_abort("%s: internal error - aborting: [%s(%d)]\n",__func__, __FILE__ , __LINE__);
    break;
  case(ecl_summary):
    if (state == analyzed)
      driver = fs->dynamic_analyzed;
    else if (state == forecast)
      driver = fs->dynamic_forecast;
    else 
      util_abort("%s: internal error - aborting: [%s(%d)]\n",__func__, __FILE__ , __LINE__);
    break;
  case(ecl_static):
    driver = fs->eclipse_static;
    break;
  default:
    util_abort("%s: fatal internal error - could not determine enkf_fs driver for object - aborting: [%s(%d)]\n",__func__, __FILE__ , __LINE__);
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
  fs_index_free(fs->index);
  free(fs);
}


void enkf_fs_fwrite_node(enkf_fs_type * enkf_fs , enkf_node_type * enkf_node , int report_step , int iens , state_enum state) {
  basic_driver_type * driver = enkf_fs_select_driver(enkf_fs , enkf_node_get_config(enkf_node) , state);
  driver->save(driver , report_step , iens , state , enkf_node); 
}


void enkf_fs_fread_node(enkf_fs_type * enkf_fs , enkf_node_type * enkf_node , int report_step , int iens , state_enum state) {
  basic_driver_type * driver = enkf_fs_select_driver(enkf_fs , enkf_node_get_config(enkf_node) , state);
  driver->load(driver , report_step , iens , state , enkf_node); 
}

bool enkf_fs_has_node(enkf_fs_type * enkf_fs , const enkf_config_node_type * config_node , int report_step , int iens , state_enum state) {
  basic_driver_type * driver = enkf_fs_select_driver(enkf_fs , config_node , state);
  return driver->has_node(driver , report_step , iens , state , enkf_config_node_get_key_ref(config_node));
}



void enkf_fs_add_index_node(enkf_fs_type * enkf_fs , int report_step , int iens , const char * kw , enkf_var_type var_type , enkf_impl_type impl_type) {
  fs_index_add_node(enkf_fs->index , report_step , iens , kw , var_type , impl_type);
}


void enkf_fs_fwrite_restart_kw_list(enkf_fs_type * enkf_fs , int report_step , int iens, restart_kw_list_type * kw_list) {
  fs_index_fwrite_restart_kw_list(enkf_fs->index , report_step , iens , kw_list);
}


void enkf_fs_fread_restart_kw_list(enkf_fs_type * enkf_fs , int report_step , int iens, restart_kw_list_type * kw_list) {
  fs_index_fread_restart_kw_list(enkf_fs->index , report_step , iens , kw_list);
}


enkf_node_type * enkf_fs_fread_alloc_node(enkf_fs_type * enkf_fs , enkf_config_node_type * config_node , int report_step , int iens , state_enum state) {
  enkf_node_type * node = enkf_node_alloc(config_node);
  enkf_node_ensure_memory(node);
  enkf_fs_fread_node(enkf_fs , node , report_step , iens , state);
  return node;
}

