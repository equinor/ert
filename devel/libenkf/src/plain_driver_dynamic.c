#include <stdlib.h>
#include <enkf_node.h>
#include <basic_driver.h>
#include <plain_driver_dynamic.h>
#include <plain_driver_common.h>
#include <fs_types.h>
#include <path_fmt.h>
#include <util.h>


/**
   The actual path to a stored node can be divided in three parts:


   /some/path/in/the/filesystem/CurrentDirectory/%03d/mem%03d/Analyzed/%s
   |<--------- 1 ------------->|<----- 2 ------>|<-------- 3 ---------->|

   1: This is root path of the enkf_fs filesystem. This is the path
      specified with the ENS_PATH configuration variable in the
      enkf_config system.

   2: The "directory" is a sub indexing under the root path. Typicall
      use of this is to differentiate between the enkf assimalition,
      various forward runs, smoother solutions and so on.

   3. The part with the %d variables in is the final storage
      hierarchy, where the first the replacement (%d,%d,%s) ->
      (report_step , iens, key) is done on run_time.

*/


struct plain_driver_dynamic_struct {
  BASIC_DRIVER_FIELDS;               /* The function pointers. */
  int             __id;              /* Internal id used for run_time checking of casts. */
  path_fmt_type * forecast_path;     /* The full forecast path (1+2+3 above) as a path_fmt instance. */ 
  path_fmt_type * analyzed_path;     /* The full analyzed path (1+2+3 above) as a path_fmt instance. */ 

  /*--------------------------------*/
  /* 
     Everything below the the line is internal information which is
     only consulted when selecting a new directory with
     plain_driver_select_directory().
  */
  
  char    	* root_path;         /* The root path of storage hierarchy - 1 above. (can be NULL, if ENS_PATH is relative). */
  char    	* forecast_fmt;      /* The part with %d specifiers for the forecast - 3 above. */
  char    	* analyzed_fmt;      /* The part with %d specifiers for the analyzed - 3 above. */
                
  /* 
     Observe that the current directory, 2 above, is only stored in
     the enkf_fs layer. Here it is only input to the
     plain_driver_select_directory() function.
  */
};







static plain_driver_dynamic_type * plain_driver_dynamic_safe_cast(void * _driver) {
  plain_driver_dynamic_type * driver = (plain_driver_dynamic_type *) _driver;

  if (driver->__id != PLAIN_DRIVER_DYNAMIC_ID) 
    util_abort("%s: internal error - cast failed - aborting \n",__func__);
  
  return driver;
}




static char * plain_driver_dynamic_alloc_filename(const plain_driver_dynamic_type * driver , int report_step , int iens , state_enum state , const char * key, bool auto_mkdir) {
  path_fmt_type * path = NULL;

  if (state == analyzed)
    path = driver->analyzed_path;
  else if (state == forecast) {
    if (report_step == 0)
      path = driver->analyzed_path;
    else
      path = driver->forecast_path;
  } else 
    util_abort("%s: state:%d is invalid \n",__func__ , state);

  return path_fmt_alloc_file(path , auto_mkdir , report_step , iens , key);
}



void plain_driver_dynamic_load_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_dynamic_type * driver = plain_driver_dynamic_safe_cast(_driver);
  {
    char * filename = plain_driver_dynamic_alloc_filename(driver , report_step , iens , state , enkf_node_get_key(node) , false);
    plain_driver_common_load_node(filename , report_step , iens , state , node);
    free(filename);
  }
}


void plain_driver_dynamic_unlink_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_dynamic_type * driver = plain_driver_dynamic_safe_cast(_driver);
  {
    char * filename = plain_driver_dynamic_alloc_filename(driver , report_step , iens , state , enkf_node_get_key(node) , false);
    util_unlink_existing(filename);
    free(filename);
  }
}


void plain_driver_dynamic_save_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_dynamic_type * driver = plain_driver_dynamic_safe_cast(_driver);
  {
    char * filename = plain_driver_dynamic_alloc_filename(driver , report_step , iens , state , enkf_node_get_key(node) , true);
    plain_driver_common_save_node(filename , report_step , iens , state , node);
    free(filename);
  }
}


/**
   Return true if we have a on-disk representation of the node.
*/

bool plain_driver_dynamic_has_node(void * _driver , int report_step , int iens , state_enum state , const char * key) {
  plain_driver_dynamic_type * driver = plain_driver_dynamic_safe_cast(_driver);
  {
    bool has_node;
    char * filename = plain_driver_dynamic_alloc_filename(driver , report_step , iens , state , key , false);
    if (util_file_exists(filename))
      has_node = true;
    else
      has_node = false;
    free(filename);
    return has_node;
  }
}



/**
   The enkf_fs layer has already made certain that this directory is different
   from the current. 
*/
void plain_driver_dynamic_select_dir(void *_driver , const char * directory) {
  plain_driver_dynamic_type * driver = plain_driver_dynamic_safe_cast(_driver);
  driver->forecast_path = plain_driver_common_realloc_path_fmt(driver->forecast_path , driver->root_path , directory , driver->forecast_fmt);
  driver->analyzed_path = plain_driver_common_realloc_path_fmt(driver->analyzed_path , driver->root_path , directory , driver->analyzed_fmt);
}



void plain_driver_dynamic_free(void *_driver) {
  plain_driver_dynamic_type * driver = plain_driver_dynamic_safe_cast(_driver);
  path_fmt_free(driver->forecast_path);
  path_fmt_free(driver->analyzed_path);
  free(driver->root_path);
  free(driver->forecast_fmt);
  free(driver->analyzed_fmt);
  free(driver);
}




/*
  The driver takes a copy of the path object, i.e. it can be deleted
  in the calling scope after calling plain_driver_dynamic_alloc().
*/
void * plain_driver_dynamic_alloc(const char * root_path , const char * forecast_fmt , const char * analyzed_fmt) {
  plain_driver_dynamic_type * driver = util_malloc(sizeof * driver , __func__);
  driver->select_dir    = plain_driver_dynamic_select_dir; 
  driver->load        	= plain_driver_dynamic_load_node;
  driver->save        	= plain_driver_dynamic_save_node;
  driver->has_node    	= plain_driver_dynamic_has_node;
  driver->free_driver 	= plain_driver_dynamic_free;
  driver->unlink_node 	= plain_driver_dynamic_unlink_node;
  driver->root_path     = util_alloc_string_copy( root_path    );
  driver->forecast_fmt  = util_alloc_string_copy( forecast_fmt );
  driver->analyzed_fmt  = util_alloc_string_copy( analyzed_fmt ); 
  driver->forecast_path = NULL;
  driver->analyzed_path = NULL;
  driver->__id = PLAIN_DRIVER_DYNAMIC_ID;
  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}


void plain_driver_dynamic_fwrite_mount_info(FILE * stream , bool read, const char * forecast_fmt , const char * analyzed_fmt) {
  util_fwrite_bool(read , stream);
  util_fwrite_int(DRIVER_DYNAMIC , stream);
  util_fwrite_int(PLAIN_DRIVER_DYNAMIC_ID , stream);
  util_fwrite_string(forecast_fmt , stream);
  util_fwrite_string(analyzed_fmt , stream);
}

/**
   The two integers from the mount info have already been read at the enkf_fs level.
*/
plain_driver_dynamic_type * plain_driver_dynamic_fread_alloc(const char * root_path , FILE * stream) {
  char * forecast_fmt = util_fread_alloc_string( stream );
  char * analyzed_fmt = util_fread_alloc_string( stream );
  plain_driver_dynamic_type * driver = plain_driver_dynamic_alloc(root_path , forecast_fmt , analyzed_fmt);
  free(forecast_fmt);
  free(analyzed_fmt);
  return driver;
}

