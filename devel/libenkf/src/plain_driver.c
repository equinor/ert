#include <stdlib.h>
#include <enkf_node.h>
#include <basic_driver.h>
#include <plain_driver.h>
#include <path_fmt.h>
#include <util.h>

#define PLAIN_DRIVER_ID 1001

struct plain_driver_struct {
  BASIC_DRIVER_FIELDS;
  int             plain_driver_id;
  path_fmt_type * path;
};





static void plain_driver_assert_cast(plain_driver_type * plain_driver) {
  if (plain_driver->plain_driver_id != PLAIN_DRIVER_ID) 
    util_abort("%s: internal error - cast failed - aborting \n",__func__);
}



void plain_driver_load_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_type * driver = (plain_driver_type *) _driver;
  plain_driver_assert_cast(driver);
  {
    char * filename = path_fmt_alloc_file(driver->path , false , report_step , iens , enkf_node_get_key_ref(node));
    FILE * stream = util_fopen(filename , "r");
    enkf_node_fread(node , stream);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_unlink_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_type * driver = (plain_driver_type *) _driver;
  plain_driver_assert_cast(driver);
  {
    char * filename = path_fmt_alloc_file(driver->path , false , report_step , iens , enkf_node_get_key_ref(node));
    util_unlink_existing(filename);
    free(filename);
  }
}


void plain_driver_save_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_type * driver = (plain_driver_type *) _driver;
  plain_driver_assert_cast(driver);
  {
    char * filename = path_fmt_alloc_file(driver->path , true , report_step , iens , enkf_node_get_key_ref(node));
    FILE * stream = util_fopen(filename , "w");
    enkf_node_fwrite(node , stream);
    fclose(stream);
    free(filename);
  }
}


/**
   Return true if we have a on-disk representation of the node.
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
  free(driver);
}



void plain_driver_README(const char * root_path) {
  char * README_file = util_alloc_full_path(root_path , "README.txt");
  util_make_path(root_path);
  {
    FILE * stream      = util_fopen(README_file , "w");
    fprintf(stream,"This is the root directory of the EnKF ensemble filesystem. All files contain one enkf_node \n");
    fprintf(stream,"instance. The files are binary, and compressed with zlib (util_fwrite_compressed).\n");
    fclose(stream);
  }
  free(README_file);
}


/*
  The driver takes a copy of the path object, i.e. it can be deleted
  in the calling scope after calling plain_driver_alloc().
*/
void * plain_driver_alloc(const char * root_path , const char * driver_path) {
  plain_driver_type * driver = malloc(sizeof * driver);
  driver->load        = plain_driver_load_node;
  driver->save        = plain_driver_save_node;
  driver->has_node    = plain_driver_has_node;
  driver->free_driver = plain_driver_free;
  driver->unlink_node = plain_driver_unlink_node;
  {
    char *path;
    if (root_path != NULL)
      path = util_alloc_full_path(root_path , driver_path);
    else
      path = util_alloc_string_copy(driver_path);
    
    driver->path = path_fmt_alloc_directory_fmt(path );
    free(path);
  }
  driver->plain_driver_id = PLAIN_DRIVER_ID;
  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}
