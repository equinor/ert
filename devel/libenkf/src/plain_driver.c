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
  if (plain_driver->plain_driver_id != PLAIN_DRIVER_ID) {
    fprintf(stderr,"%s: internal error - cast failed - aborting \n",__func__);
    abort();
  }
}


void plain_driver_load_node(void * _driver , int report_step , int iens , bool analyzed , enkf_node_type * node) {
  plain_driver_type * driver = (plain_driver_type *) _driver;
  plain_driver_assert_cast(driver);
  {
    char * filename;
    FILE * stream = util_fopen(filename , "r");
    enkf_node_fread(node , stream);
    fclose(stream);
  }
}


void plain_driver_save_node(void * _driver , int report_step , int iens , bool analyzed , enkf_node_type * node) {
  plain_driver_type * driver = (plain_driver_type *) _driver;
  plain_driver_assert_cast(driver);
  {
    char * filename;
    FILE * stream = util_fopen(filename , "r");
    enkf_node_fwrite(node , stream);
    fclose(stream);
  }
}


void plain_driver_swapout_node(void * _driver , int report_step , int iens , bool analyzed , enkf_node_type * node) {
  plain_driver_type * driver = (plain_driver_type *) _driver;
  plain_driver_assert_cast(driver);
  {
    char * filename;
    FILE * stream = util_fopen(filename , "r");
    enkf_node_swapout(node , stream);
    fclose(stream);
  }
}


void plain_driver_swapin_node(void * _driver , int report_step , int iens , bool analyzed , enkf_node_type * node) {
  plain_driver_type * driver = (plain_driver_type *) _driver;
  plain_driver_assert_cast(driver);
  {
    char * filename;
    FILE * stream = util_fopen(filename , "r");
    enkf_node_swapin(node , stream);
    fclose(stream);
  }
}


void plain_driver_free(void *_driver) {
  plain_driver_type * driver = (plain_driver_type *) _driver;
  plain_driver_assert_cast(driver);
  path_fmt_free(driver->path);
  free(driver);
}


/*
  The driver takes a copy of the path object, i.e. it can be deleted
  in the calling scope after calling plain_driver_alloc().
*/
void * plain_driver_alloc(const char * path) {
  plain_driver_type * driver = malloc(sizeof * driver);
  driver->load        = plain_driver_load_node;
  driver->save        = plain_driver_save_node;
  driver->swapout     = plain_driver_swapout_node;
  driver->swapin      = plain_driver_swapin_node;
  driver->free_driver = plain_driver_free;
  driver->path        = path_fmt_alloc_directory_fmt(path , true);
  driver->plain_driver_id = PLAIN_DRIVER_ID;
  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}
