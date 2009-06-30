#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <util.h>
#include <path_fmt.h>
#include <fs_types.h>
#include <buffer.h>
#include <basic_driver.h>
#include <plain_driver.h>
#include <plain_driver_index.h>
#include <plain_driver_common.h>



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


struct plain_driver_struct {
  BASIC_DRIVER_FIELDS;
  int             plain_driver_id;
  path_fmt_type * path;
  
  /* ---------------------------: The different parts of the path variable is documented in plain_driver_dynamic. */
  char          * root_path;
  char          * fmt_string;
};



static void plain_driver_assert_cast(plain_driver_type * plain_driver) {
  if (plain_driver->plain_driver_id != PLAIN_DRIVER_ID) 
    util_abort("%s: internal error - cast failed - aborting \n",__func__);
}


static plain_driver_type * plain_driver_safe_cast( void * __driver) {
  plain_driver_type * driver = (plain_driver_type *) __driver;
  plain_driver_assert_cast(driver);
  return driver;
}



static void plain_driver_load_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_type * driver = plain_driver_safe_cast( _driver );
  buffer_type * buffer       = buffer_alloc(100);
  {
    char * filename      = path_fmt_alloc_file(driver->path , false , report_step , iens , enkf_node_get_key(node));
    buffer_fread_realloc( buffer , filename );
    buffer_fskip_time_t( buffer );
    enkf_node_load(node , buffer , report_step, iens , state);
    free(filename);
  }
  buffer_free( buffer );
}




static void plain_driver_save_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_type * driver = (plain_driver_type *) _driver;
  plain_driver_assert_cast(driver);
  {
    bool   internal_state = true;
    bool   data_written;
    buffer_type * buffer = buffer_alloc(100);
    buffer_fwrite_time_t( buffer , time(NULL));
    data_written 	 = enkf_node_store(node , buffer , internal_state , report_step , iens , state);  /* <- Even this could (should) be done at the enkf_fs level. */
    
    if (data_written) {
      char * filename = path_fmt_alloc_file(driver->path , true , report_step , iens , enkf_node_get_key(node));
      buffer_store( buffer , filename );
      free(filename);
    }
    
    buffer_free( buffer );
  }
}


void plain_driver_unlink_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_type * driver = (plain_driver_type *) _driver;
  plain_driver_assert_cast(driver);
  {
    char * filename = path_fmt_alloc_file(driver->path , true , report_step , iens , enkf_node_get_key(node));
    util_unlink_existing(filename);
    free(filename);
  }
}


/**
   Observe that the semantics is fundamentally different between
   plain_driver_paramater_has_node, and plain_driver_load_node:

   * When (trying to) load a node the function will try previous report steps
     all the way back to the first report step.

   * The has_node function will _not_ go back to earlier report steps, but
     instead return false if the report_step we ask for is not present.
*/

bool plain_driver_has_node(void * _driver , int report_step , int iens , state_enum state , const char * key) {
  plain_driver_type * driver = (plain_driver_type *) _driver;
  plain_driver_assert_cast(driver);
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




void plain_driver_free(void *_driver) {
  plain_driver_type * driver = (plain_driver_type *) _driver;
  plain_driver_assert_cast(driver);
  path_fmt_free(driver->path);
  free( driver->fmt_string );
  util_safe_free( driver->root_path );
  free(driver);
}


/**
   The enkf_fs layer has already made certain that this directory is different
   from the current. 
*/
void plain_driver_select_dir(void *_driver , const char * directory) {
  plain_driver_type * driver = plain_driver_safe_cast(_driver);
  driver->path      = plain_driver_common_realloc_path_fmt(driver->path , driver->root_path , directory , driver->fmt_string);
}





/**
  The driver takes a copy of the path object, i.e. it can be deleted
  in the calling scope after calling plain_driver_alloc().

  This is where the various function pointers are initialized.
*/
void * plain_driver_alloc(const char * root_path , const char * fmt) {
  plain_driver_type * driver = malloc(sizeof * driver);

  driver->load        	= plain_driver_load_node;
  driver->save        	= plain_driver_save_node;
  driver->free_driver 	= plain_driver_free;
  driver->unlink_node 	= plain_driver_unlink_node;
  driver->has_node    	= plain_driver_has_node;
  driver->select_dir    = plain_driver_select_dir;
  driver->root_path     = util_alloc_string_copy( root_path );
  driver->fmt_string    = util_alloc_string_copy( fmt );
  driver->path          = NULL;    /* This is set by calls to select_dir */
  driver->plain_driver_id = PLAIN_DRIVER_ID;
  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}




void plain_driver_fwrite_mount_info(FILE * stream , fs_driver_type driver_type , bool read , const char * fmt ) {
  util_fwrite_bool(read , stream);
  util_fwrite_int(driver_type , stream);
  util_fwrite_int(PLAIN_DRIVER_ID , stream);
  util_fwrite_string(fmt , stream);
}

/**
   The two integers from the mount info have already been read at the enkf_fs level.
*/
plain_driver_type * plain_driver_fread_alloc(const char * root_path , FILE * stream) {
  char * fmt = util_fread_alloc_string( stream );
  plain_driver_type * driver = plain_driver_alloc(root_path , fmt );
  free(fmt);
  return driver;
}


