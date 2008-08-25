#include <stdlib.h>
#include <enkf_node.h>
#include <basic_driver.h>
#include <plain_driver_static.h>
#include <ecl_static_kw.h>
#include <path_fmt.h>
#include <util.h>

#define PLAIN_DRIVER_STATIC_ID 1005


struct plain_driver_static_struct {
  BASIC_DRIVER_FIELDS;
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

static plain_driver_static_type * plain_driver_static_init(void *_driver) {
  plain_driver_static_type * driver = (plain_driver_static_type *) _driver;
  if (driver->plain_driver_static_id != PLAIN_DRIVER_STATIC_ID) 
    util_abort("%s: internal error - cast failed - aborting \n",__func__);
  return driver;
}





char * plain_driver_static_alloc_filename(plain_driver_static_type * driver, int report_step , int iens , state_enum state , enkf_node_type * node , bool auto_mkdir) {
  ecl_static_kw_type * ecl_static = enkf_node_value_ptr( node );
  ecl_static_kw_assert_type(ecl_static);
  return path_fmt_alloc_file(driver->path , auto_mkdir , report_step , iens , enkf_node_get_key_ref(node) , ecl_static_kw_get_counter(ecl_static));
}




void plain_driver_static_load_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_init(_driver);
  {
    char * filename = plain_driver_static_alloc_filename(driver , report_step , iens , state , node , false);
    FILE * stream   = util_fopen(filename , "r");
    enkf_node_fread(node , stream);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_static_unlink_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_init(_driver);
  {
    char * filename = plain_driver_static_alloc_filename(driver , report_step , iens , state , node , false);
    FILE * stream   = util_fopen(filename , "w");
    util_unlink_existing(filename);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_static_save_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_init(_driver);
  {
    char * filename = plain_driver_static_alloc_filename(driver , report_step , iens , state , node , true);
    FILE * stream   	 = util_fopen(filename , "w");
    enkf_node_fwrite(node , stream);
    fclose(stream);
    free(filename);
  }
}


/**
   Return true if we have a on-disk representation of the node.
*/

bool plain_driver_static_has_node(void * _driver , int report_step , int iens , state_enum state , const char * key) {
  util_abort("%s: internal error - the filesystem does not support query on static nodes ... \n",__func__);
  return false;
}





void plain_driver_static_free(void *_driver) {
  plain_driver_static_type * driver = plain_driver_static_init(_driver);
  path_fmt_free(driver->path);
  free(driver);
}



void * plain_driver_static_alloc(const char * root_path , const char * driver_path) {
  plain_driver_static_type * driver = malloc(sizeof * driver);
  driver->load        = plain_driver_static_load_node;
  driver->save        = plain_driver_static_save_node;
  driver->has_node    = plain_driver_static_has_node;
  driver->free_driver = plain_driver_static_free;
  driver->unlink_node = plain_driver_static_unlink_node;
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
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}
