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
  if (plain_driver_parameter->plain_driver_parameter_id != PLAIN_DRIVER_ID) 
    util_abort("%s: internal error - cast failed - aborting \n",__func__);
}



/**
   The convention is that if we ask for the analyzed, we get the
   report step according to input, if we ask for the forecast the
   report step is set one back.

   This means that the function will fail hard if we ask for the
   forecast at report_step == 0 - which is maybe fair enough.
*/

static int __get_report_step(int report_step , state_enum state) {
  if (state == analyzed)
    return report_step;
  else if (state == forecast) {
    if (report_step == 0) 
      util_abort("%s: sorry at report_step:0 there is no forecast - aborting \n",__func__);
    return report_step - 1;
  } else {
    util_abort("%s state:%d - internal error - aborting \n",__func__ , state);
    return -1;  /* Shut up the compiler */
  }
}


/**
   Observe that if we do not find the filename we are looking for, we
   seek backwards through the report numbers, all the way back to
   report_nr 0. The direct motivation for this functionality is the
   following situation:

   1. We do a spin-up from report 0 to report R1.

   2. We start the assimulation from R1, then we have to go all the
      way back to report 0 to get hold of the parameter.
*/

void plain_driver_parameter_load_node(void * _driver , int _report_step , int iens , state_enum state , enkf_node_type * node) {
  int report_step = __get_report_step(_report_step , state);
  plain_driver_parameter_type * driver = (plain_driver_parameter_type *) _driver;
  plain_driver_parameter_assert_cast(driver);
  {
    char * filename;
    filename = path_fmt_alloc_file(driver->path , false , report_step , iens , enkf_node_get_ensfile_ref(node));
    while (!util_file_exists(filename)) {
      report_step--;
      if (report_step < 0) 
	util_abort("%s can not find any stored item for ??? \n",__func__);
      else {
	free(filename);
	filename = path_fmt_alloc_file(driver->path , false , report_step , iens , enkf_node_get_ensfile_ref(node));
      }
    }
    {
      FILE * stream   = util_fopen(filename , "r");
      enkf_node_fread(node , stream);
      fclose(stream);
    }
    free(filename);
  }
}


void plain_driver_parameter_save_node(void * _driver , int _report_step , int iens , state_enum state , enkf_node_type * node) {
  int report_step = __get_report_step(_report_step , state);
  plain_driver_parameter_type * driver = (plain_driver_parameter_type *) _driver;
  plain_driver_parameter_assert_cast(driver);
  {
    char * filename = path_fmt_alloc_file(driver->path , true , report_step , iens , enkf_node_get_ensfile_ref(node));
    FILE * stream = util_fopen(filename , "w");
    enkf_node_fwrite(node , stream);
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
  driver->free_driver = plain_driver_parameter_free;
  {
    char *path;
    if (root_path != NULL)
      path = util_alloc_full_path(root_path , driver_path);
    else
      path = util_alloc_string_copy(driver_path);
    
    driver->path        = path_fmt_alloc_directory_fmt(path);
    free(path);
  }
  driver->plain_driver_parameter_id = PLAIN_DRIVER_ID;
  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}
