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
  path_fmt_type * path;                     
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






void plain_driver_static_load_node(void * _driver , int report_step , int iens , state_enum state , int static_counter , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_safe_cast(_driver);
  {
    char * filename = path_fmt_alloc_file(driver->path , false , report_step , iens , enkf_node_get_key(node) , static_counter);
    plain_driver_common_load_node(filename , report_step , iens , state , node);
    free(filename);
  }
}


void plain_driver_static_unlink_node(void * _driver , int report_step , int iens , state_enum state , int static_counter , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_safe_cast(_driver);
  {
    char * filename = path_fmt_alloc_file(driver->path , false , report_step , iens , enkf_node_get_key(node) , static_counter);
    FILE * stream   = util_fopen(filename , "w");
    util_unlink_existing(filename);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_static_save_node(void * _driver , int report_step , int iens , state_enum state , int static_counter , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_safe_cast(_driver);
  {
    char * filename      = path_fmt_alloc_file(driver->path , true , report_step , iens , enkf_node_get_key(node) , static_counter);
    plain_driver_common_save_node(filename , report_step , iens , state , node);
    free(filename);
  }
}


/**
   Return true if we have a on-disk representation of the node.
*/

bool plain_driver_static_has_node(void * _driver , int report_step , int iens , state_enum state , int static_counter , const char * key) {
  plain_driver_static_type * driver = plain_driver_static_safe_cast(_driver);
  {
    bool has_node;
    char * filename = path_fmt_alloc_file(driver->path , true , report_step , iens , key , static_counter);
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
  path_fmt_free(driver->path);
  free(driver);
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
  {
    char *path;

    /**
       The format is:
       
       [root_path]/driver_path/<STATIC-KW>/<INTEGER>
    */

    if (root_path != NULL) 
      path = util_alloc_sprintf("%s%c%s%c%s%c%s" , root_path , UTIL_PATH_SEP_CHAR , driver_path , UTIL_PATH_SEP_CHAR , "%s" , UTIL_PATH_SEP_CHAR , "%d");
    else
      path = util_alloc_sprintf("%s%c%s%c%s" , driver_path , UTIL_PATH_SEP_CHAR , "%s" , UTIL_PATH_SEP_CHAR , "%d");
    
    driver->path = path_fmt_alloc_path_fmt( path );
    free(path);
  }
  driver->plain_driver_static_id = PLAIN_DRIVER_STATIC_ID;
  {
    basic_static_driver_type * basic_driver = (basic_static_driver_type *) driver;
    basic_static_driver_init(basic_driver);
    return basic_driver;
  }
}


void plain_driver_static_fwrite_mount_info(FILE * stream , const char * fmt) {
  util_fwrite_int(STATIC_DRIVER , stream);
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

