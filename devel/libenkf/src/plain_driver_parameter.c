#include <stdlib.h>
#include <enkf_node.h>
#include <basic_driver.h>
#include <plain_driver_parameter.h>
#include <path_fmt.h>
#include <util.h>

#define PLAIN_DRIVER_ID 1003

struct plain_driver_parameter_struct {
  BASIC_DRIVER_FIELDS;
  int             plain_driver_parameter_id;
  path_fmt_type * path;
};


static void plain_driver_parameter_assert_cast(plain_driver_parameter_type * plain_driver_parameter) {
  if (plain_driver_parameter->plain_driver_parameter_id != PLAIN_DRIVER_ID) {
    fprintf(stderr,"%s: internal error - cast failed - aborting \n",__func__);
    abort();
  }
}



/**
   The convention is that if we ask for the analyzed, we get the
   report step according to input, if we ask for the forecast the
   report step is set one back.
*/

static int __get_report_step(int report_step , state_enum state) {
  if (state == analyzed)
    return report_step;
  else if (state == forecast)
    return report_step - 1;
  else {
    fprintf(stderr,"%s state:%d - internal error - aborting \n",__func__ , state);
    abort();
  }
}

void plain_driver_parameter_load_node(void * _driver , int _report_step , int iens , state_enum state , enkf_node_type * node) {
  int report_step = __get_report_step(_report_step , state);
  plain_driver_parameter_type * driver = (plain_driver_parameter_type *) _driver;
  plain_driver_parameter_assert_cast(driver);
  {
    char * filename = path_fmt_alloc_file(driver->path , report_step , iens , enkf_node_get_ensfile_ref(node));
    FILE * stream = util_fopen(filename , "r");
    enkf_node_fread(node , stream);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_parameter_save_node(void * _driver , int _report_step , int iens , state_enum state , enkf_node_type * node) {
  int report_step = __get_report_step(_report_step , state);
  plain_driver_parameter_type * driver = (plain_driver_parameter_type *) _driver;
  plain_driver_parameter_assert_cast(driver);
  {
    char * filename = path_fmt_alloc_file(driver->path , report_step , iens , enkf_node_get_ensfile_ref(node));
    FILE * stream = util_fopen(filename , "w");
    enkf_node_fwrite(node , stream);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_parameter_swapout_node(void * _driver , int _report_step , int iens , state_enum state , enkf_node_type * node) {
  int report_step = __get_report_step(_report_step , state);
  plain_driver_parameter_type * driver = (plain_driver_parameter_type *) _driver;
  plain_driver_parameter_assert_cast(driver);
  if (!enkf_node_swapped(node)) {
    char * filename = path_fmt_alloc_file(driver->path , report_step , iens , enkf_node_get_ensfile_ref(node));
    FILE * stream = util_fopen(filename , "w");
    enkf_node_swapout(node , stream);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_parameter_swapin_node(void * _driver , int _report_step , int iens , state_enum state , enkf_node_type * node) {
  int report_step = __get_report_step(_report_step , state);
  plain_driver_parameter_type * driver = (plain_driver_parameter_type *) _driver;
  plain_driver_parameter_assert_cast(driver);
  {
    char * filename = path_fmt_alloc_file(driver->path , report_step , iens , enkf_node_get_ensfile_ref(node));
    FILE * stream = util_fopen(filename , "r");
    enkf_node_swapin(node , stream);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_parameter_free(void *_driver) {
  plain_driver_parameter_type * driver = (plain_driver_parameter_type *) _driver;
  plain_driver_parameter_assert_cast(driver);
  path_fmt_free(driver->path);
  free(driver);
}


/*
  The driver takes a copy of the path object, i.e. it can be deleted
  in the calling scope after calling plain_driver_parameter_alloc().
*/
void * plain_driver_parameter_alloc(const char * root_path , const char * driver_path) {
  plain_driver_parameter_type * driver = malloc(sizeof * driver);
  driver->load        = plain_driver_parameter_load_node;
  driver->save        = plain_driver_parameter_save_node;
  driver->swapout     = plain_driver_parameter_swapout_node;
  driver->swapin      = plain_driver_parameter_swapin_node;
  driver->free_driver = plain_driver_parameter_free;
  {
    char *path;
    if (root_path != NULL)
      path = util_alloc_full_path(root_path , driver_path);
    else
      path = util_alloc_string_copy(driver_path);
    
    driver->path        = path_fmt_alloc_directory_fmt(path , true);
    free(path);
  }
  driver->plain_driver_parameter_id = PLAIN_DRIVER_ID;
  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}
