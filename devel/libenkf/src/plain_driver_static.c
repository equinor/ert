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
  int             plain_driver_static_id;
  path_fmt_type * path;
};






static void plain_driver_static_assert_cast(plain_driver_static_type * plain_driver_static) {
  if (plain_driver_static->plain_driver_static_id != PLAIN_DRIVER_STATIC_ID) 
    util_abort("%s: internal error - cast failed - aborting \n",__func__);
}


static plain_driver_static_type * plain_driver_static_init(void *_driver) {
  plain_driver_static_type * driver = (plain_driver_static_type *) _driver;
  plain_driver_static_assert_cast(driver);
  return driver;
}


char * plain_driver_static_alloc_filename(plain_driver_static_type * driver, int report_step , int iens , state_enum state , enkf_node_type * node) {
  ecl_static_kw_type * ecl_static = enkf_node_value_ptr( node );
  ecl_static_kw_assert_type(ecl_static);
  {
    /*
      Observe that the function path_fmt_alloc_file() assumes that the final argument (the filename)
      is a string.
    */
    char * counter_str = util_alloc_sprintf("%d" , ecl_static_kw_get_counter(ecl_static));
    char * filename 	 = path_fmt_alloc_file(driver->path , false , report_step , iens , enkf_node_get_ensfile_ref(node) , counter_str);
    free(counter_str);
    return filename;
  }
}



void plain_driver_static_load_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_init(_driver);
  {
    char * filename = plain_driver_static_alloc_filename(driver , report_step , iens , state , node);
    FILE * stream   	 = util_fopen(filename , "r");
    enkf_node_fread(node , stream);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_static_unlink_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_init(_driver);
  {
    char * filename = plain_driver_static_alloc_filename(driver , report_step , iens , state , node);
    FILE * stream   	 = util_fopen(filename , "w");
    util_unlink_existing(filename);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_static_save_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_init(_driver);
  {
    char * filename = plain_driver_static_alloc_filename(driver , report_step , iens , state , node);
    FILE * stream   	 = util_fopen(filename , "w");
    enkf_node_fwrite(node , stream);
    fclose(stream);
    free(filename);
  }
}


/**
   Return true if we have a on-disk representation of the node.
*/

bool plain_driver_static_has_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = plain_driver_static_init(_driver);
  {
    bool has_node;
    char * filename = plain_driver_static_alloc_filename(driver , report_step , iens , state , node);
    if (util_file_exists(filename))
      has_node = true;
    else
      has_node = false;
    free(filename);
    return has_node;
  }
}




void plain_driver_static_free(void *_driver) {
  plain_driver_static_type * driver = plain_driver_static_init(_driver);
  path_fmt_free(driver->path);
  free(driver);
}



void plain_driver_static_README(const char * root_path) {
  char * README_file = util_alloc_full_path(root_path , "README.txt");
  util_make_path(root_path);
  {
    FILE * stream  = util_fopen(README_file , "w");
    fprintf(stream,"This is the root directory of the EnKF ensemble filesystem. All files contain one enkf_node \n");
    fprintf(stream,"instance. The files are binary, and compressed with zlib (util_fwrite_compressed).\n");
    fclose(stream);
  }
  free(README_file);
}


/*
  The driver takes a copy of the path object, i.e. it can be deleted
  in the calling scope after calling plain_driver_static_alloc().
*/
void * plain_driver_static_alloc(const char * root_path , const char * driver_path) {
  plain_driver_static_type * driver = malloc(sizeof * driver);
  driver->load        = plain_driver_static_load_node;
  driver->save        = plain_driver_static_save_node;
  driver->has_node    = plain_driver_static_has_node;
  driver->free_driver = plain_driver_static_free;
  driver->unlink_node = plain_driver_static_unlink_node;
  {
    char *path;

    if (root_path != NULL) 
      path = util_alloc_sprintf("%s%c%s%c%s" , root_path , UTIL_PATH_SEP_CHAR , driver_path , UTIL_PATH_SEP_CHAR , "%s" );
    else
      path = util_alloc_sprintf("%s%c%s" , driver_path , UTIL_PATH_SEP_CHAR , "%s");
    
    driver->path = path_fmt_alloc_directory_fmt( path );
    free(path);
  }
  driver->plain_driver_static_id = PLAIN_DRIVER_STATIC_ID;
  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}
