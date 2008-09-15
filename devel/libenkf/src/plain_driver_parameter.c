#include <stdlib.h>
#include <enkf_node.h>
#include <basic_driver.h>
#include <plain_driver_parameter.h>
#include <fs_types.h>
#include <path_fmt.h>
#include <util.h>



struct plain_driver_parameter_struct {
  BASIC_DRIVER_FIELDS;
  int             plain_driver_parameter_id;
  path_fmt_type * path;
};


static void plain_driver_parameter_assert_cast(plain_driver_parameter_type * plain_driver_parameter) {
  if (plain_driver_parameter->plain_driver_parameter_id != PLAIN_DRIVER_PARAMETER_ID) 
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
      return 0;  /* Time step zero is special - we do not differentiate between forecast and analyzed. */
    else
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
    filename = path_fmt_alloc_file(driver->path , false , report_step , iens , enkf_node_get_key_ref(node));
    while (!util_file_exists(filename)) {
      report_step--;
      if (report_step < 0) 
	util_abort("%s: can not find any stored item for key:%s. Forgot to initialize ensemble ??? \n",__func__ , enkf_node_get_key_ref(node));
      else {
	free(filename);
	filename = path_fmt_alloc_file(driver->path , false , report_step , iens , enkf_node_get_key_ref(node));
      }
    }
    {
      FILE * stream   = util_fopen(filename , "r");
      enkf_node_fread(node , stream , report_step , state);
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
    char * filename = path_fmt_alloc_file(driver->path , true , report_step , iens , enkf_node_get_key_ref(node));
    FILE * stream = util_fopen(filename , "w");
    bool   data_written = enkf_node_fwrite(node , stream , report_step , state);
    fclose(stream);
    if (!data_written)
      unlink(filename);
    free(filename);
  }
}


void plain_driver_parameter_unlink_node(void * _driver , int _report_step , int iens , state_enum state , enkf_node_type * node) {
  int report_step = __get_report_step(_report_step , state);
  plain_driver_parameter_type * driver = (plain_driver_parameter_type *) _driver;
  plain_driver_parameter_assert_cast(driver);
  {
    char * filename = path_fmt_alloc_file(driver->path , true , report_step , iens , enkf_node_get_key_ref(node));
    util_unlink_existing(filename);
    free(filename);
  }
}


/**
   Observe that the semantics is fundamentally different between
   plain_driver_paramater_has_node, and plain_driver_parameter_load_node:

   * When (trying to) load a node the function will try previous report steps
     all the way back to the first report step.

   * The has_node function will _not_ go back to earlier report steps, but
     instead return false if the report_step we ask for is not present.
*/

bool plain_driver_parameter_has_node(void * _driver , int _report_step , int iens , state_enum state , const char * key) {
  int report_step = __get_report_step(_report_step , state);
  plain_driver_parameter_type * driver = (plain_driver_parameter_type *) _driver;
  plain_driver_parameter_assert_cast(driver);
  {
    bool has_node;
    char * filename = path_fmt_alloc_file(driver->path , true , report_step , iens , key);
    if (util_file_exists(filename))
      has_node = true;
    else
      has_node = false;
    free(filename);
    return has_node;
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
  driver->unlink_node = plain_driver_parameter_unlink_node;
  driver->has_node    = plain_driver_parameter_has_node;
  {
    char *path;
    if (root_path != NULL)
      path = util_alloc_full_path(root_path , driver_path);
    else
      path = util_alloc_string_copy(driver_path);
    
    driver->path        = path_fmt_alloc_directory_fmt(path);
    free(path);
  }
  driver->plain_driver_parameter_id = PLAIN_DRIVER_PARAMETER_ID;
  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}



void plain_driver_parameter_fwrite_mount_info(FILE * stream , const char * fmt ) {
  util_fwrite_int(PARAMETER_DRIVER , stream);
  util_fwrite_int(PLAIN_DRIVER_PARAMETER_ID , stream);
  util_fwrite_string(fmt , stream);
}

/**
   The two integers from the mount info have already been read at the enkf_fs level.
*/
plain_driver_parameter_type * plain_driver_parameter_fread_alloc(const char * root_path , FILE * stream) {
  char * fmt = util_fread_alloc_string( stream );
  plain_driver_parameter_type * driver = plain_driver_parameter_alloc(root_path , fmt );
  free(fmt);
  return driver;
}
