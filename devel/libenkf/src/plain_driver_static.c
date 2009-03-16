#include <stdlib.h>
#include <enkf_node.h>
#include <basic_driver.h>
#include <plain_driver_static.h>
#include <ecl_static_kw.h>
#include <path_fmt.h>
#include <fs_types.h>
#include <util.h>
#include <plain_driver_common.h>



struct plain_driver_static_struct {
  BASIC_STATIC_DRIVER_FIELDS;
  int             plain_driver_static_id;   /* See documentation in plain_driver.c */
  path_fmt_type * path_fmt;

  /* ---------------------------: The different parts of the path variable is documented in plain_driver_dynamic. */
  char          * root_path;
  char          * fmt;
};



/**
   This file implements the plain (filesystem) driver for the static
   keywords. In enkf_fs api a node is specified by the following
   coordinates:

    1. The report number.
    2. Whether we want the analyzed value or the forecast value.
    3. The enkf member nr.
 
   Unfortunately it turns out this is not enough to uniquely identify
   a static keyword, for instance the keywords containing information
   about AQUIFERS can occure several times in one restart file,
   therefor we must have an additional counter to uniquely identify a
   static keyword. This driver queries the ecl_static_kw instance to
   get this counter.
   
   Yes - this ugly situation arose a loooong time after the API was
   orignally designed.
*/
   


/**
   Casts a void pointer to a plain_driver_static instance (checks the
   cast), and returns a typed pointer (it is the same adress as input).
*/

static plain_driver_static_type * plain_driver_static_safe_cast(void *_driver) {
  plain_driver_static_type * driver = (plain_driver_static_type *) _driver;
  if (driver->plain_driver_static_id != PLAIN_DRIVER_STATIC_ID) 
    util_abort("%s: internal error - cast failed - aborting \n",__func__);
  return driver;
}


static char * plain_driver_static_alloc_filename__(plain_driver_static_type * driver , bool auto_mkdir , int report_step , int iens , const char * node_key) {
  return path_fmt_alloc_file(driver->path_fmt , auto_mkdir , report_step , iens , node_key);
}



/**
   Observe that all these functions use the path_fmt_alloc_file()
   functions instead of the path_fmt_alloc_file() function. That is
   because the static format string specifies the full filename, and
   not only the directory part.
*/

void plain_driver_static_load_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_safe_cast(_driver);
  {
    const char * node_key = enkf_node_get_key( node );
    char * filename = plain_driver_static_alloc_filename__(driver , false , report_step , iens , node_key);
    plain_driver_common_load_node(filename , report_step , iens , state , node);
    free(filename);
  }
}


void plain_driver_static_unlink_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_safe_cast(_driver);
  {
    const char * node_key = enkf_node_get_key( node );
    char * filename = plain_driver_static_alloc_filename__(driver , false , report_step , iens , node_key);
    util_unlink_existing(filename);
    free(filename);
  }
}


void plain_driver_static_save_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_safe_cast(_driver);
  {
    const char * node_key = enkf_node_get_key( node );
    char * filename = plain_driver_static_alloc_filename__(driver , true , report_step , iens , node_key);
    plain_driver_common_save_node(filename , report_step , iens , state , node);
    free(filename);
  }
}


/**
   Return true if we have a on-disk representation of the node.
*/

bool plain_driver_static_has_node(void * _driver , int report_step , int iens , state_enum state , const char * key) {
  plain_driver_static_type * driver = plain_driver_static_safe_cast(_driver);
  {
    bool has_node;
    char * filename = plain_driver_static_alloc_filename__(driver , false , report_step , iens , key);
    if (util_file_exists(filename))
      has_node = true;
    else
      has_node = false;
    free(filename);

    /* The static counter is hardcoded to zero from the calling scope - that is why we abort. */
    util_abort("%s: sorry function not implemented for static nodes.\n",__func__);
    return has_node;
  }
}




void plain_driver_static_free(void *_driver) {
  plain_driver_static_type * driver = plain_driver_static_safe_cast(_driver);
  path_fmt_free(driver->path_fmt);
  util_safe_free( driver->root_path );
  util_safe_free( driver->fmt );
  free(driver);
}

void plain_driver_static_select_dir(void *_driver , const char * directory) {
  plain_driver_static_type * driver = plain_driver_static_safe_cast(_driver);
  driver->path_fmt = plain_driver_common_realloc_path_fmt(driver->path_fmt , driver->root_path , directory , driver->fmt);
}




void * plain_driver_static_alloc(const char * root_path , const char * driver_path) {
  plain_driver_static_type * driver = util_malloc(sizeof * driver, __func__);
  driver->load        	= plain_driver_static_load_node;
  driver->save        	= plain_driver_static_save_node;
  driver->has_node    	= plain_driver_static_has_node;
  driver->free_driver 	= plain_driver_static_free;
  driver->unlink_node 	= plain_driver_static_unlink_node;
  driver->load_ensemble = NULL;
  driver->load_ts       = NULL;
  driver->save_ensemble = NULL;
  driver->save_ts       = NULL;
  driver->select_dir    = plain_driver_static_select_dir;
  driver->root_path     = util_alloc_string_copy( root_path );
  driver->fmt           = util_alloc_string_copy( driver_path );
  driver->path_fmt      = NULL;
  driver->plain_driver_static_id = PLAIN_DRIVER_STATIC_ID;
  {
    basic_driver_static_type * basic_driver = (basic_driver_static_type *) driver;
    basic_driver_static_init(basic_driver);
    return basic_driver;
  }
}


void plain_driver_static_fwrite_mount_info(FILE * stream , bool read , const char * fmt) {
  util_fwrite_bool(read , stream);
  util_fwrite_int(DRIVER_STATIC , stream);
  util_fwrite_int(PLAIN_DRIVER_STATIC_ID , stream);
  util_fwrite_string(fmt , stream);
}

/**
   The two integers from the mount info have already been read at the enkf_fs level.
*/
plain_driver_static_type * plain_driver_static_fread_alloc(const char * root_path , FILE * stream) {
  char * fmt = util_fread_alloc_string( stream );
  plain_driver_static_type * driver = plain_driver_static_alloc(root_path , fmt );
  free(fmt);
  return driver;
}

