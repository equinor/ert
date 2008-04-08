#include <stdlib.h>
#include <enkf_node.h>
#include <basic_driver.h>
#include <plain_driver_static.h>
#include <path_fmt.h>
#include <util.h>
#include <ecl_static_kw.h>

#define PLAIN_DRIVER_STATIC_ID 1002

struct plain_driver_static_struct {
  BASIC_DRIVER_FIELDS;
  int             plain_driver_static_id;
  path_fmt_type * path;
};


static void plain_driver_static_assert_cast(const plain_driver_static_type * plain_driver_static) {
  if (plain_driver_static->plain_driver_static_id != PLAIN_DRIVER_STATIC_ID) {
    fprintf(stderr,"%s: internal error - cast failed - aborting \n",__func__);
    abort();
  }
}

static void plain_driver_static_assert_node(const enkf_node_type * node) {
  if (enkf_node_get_var_type(node) != ecl_static) {
    fprintf(stderr,"%s: can only take nodes of type ecl_static - aborting \n",__func__);
    abort();
  }
}


static const char * __get_filename(const enkf_node_type * node) {
  return enkf_node_get_ensfile_ref(node);
}


void plain_driver_static_load_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = (plain_driver_static_type *) _driver;
  plain_driver_static_assert_cast(driver);
  plain_driver_static_assert_node(node);
  {
    char * filename = path_fmt_alloc_file(driver->path , report_step , iens , __get_filename(node));
    FILE * stream = util_fopen(filename , "r");
    enkf_node_fread(node , stream);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_static_save_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = (plain_driver_static_type *) _driver;
  plain_driver_static_assert_cast(driver);
  plain_driver_static_assert_node(node);
  {
    char * filename = path_fmt_alloc_file(driver->path , report_step , iens , __get_filename(node));
    FILE * stream = util_fopen(filename , "w");
    enkf_node_fwrite(node , stream);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_static_swapout_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = (plain_driver_static_type *) _driver;
  plain_driver_static_assert_cast(driver);
  plain_driver_static_assert_node(node);
  if (!enkf_node_swapped(node)) {
    char * filename = path_fmt_alloc_file(driver->path , report_step , iens , __get_filename(node));
    FILE * stream = util_fopen(filename , "w");
    enkf_node_swapout(node , stream);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_static_swapin_node(void * _driver , int report_step , int iens , state_enum state , enkf_node_type * node) {
  plain_driver_static_type * driver = (plain_driver_static_type *) _driver;
  plain_driver_static_assert_cast(driver);
  plain_driver_static_assert_node(node);
  {
    char * filename = path_fmt_alloc_file(driver->path , report_step , iens , __get_filename(node));
    FILE * stream = util_fopen(filename , "r");
    enkf_node_swapin(node , stream);
    fclose(stream);
    free(filename);
  }
}


void plain_driver_static_free(void *_driver) {
  plain_driver_static_type * driver = (plain_driver_static_type *) _driver;
  plain_driver_static_assert_cast(driver);
  path_fmt_free(driver->path);
  free(driver);
}


/*
  The driver takes a copy of the path object, i.e. it can be deleted
  in the calling scope after calling plain_driver_static_alloc().
*/
void * plain_driver_static_alloc(const char * root_path , const char * driver_path) {
  plain_driver_static_type * driver = malloc(sizeof * driver);
  driver->load        = plain_driver_static_load_node;
  driver->save        = plain_driver_static_save_node;
  driver->swapout     = plain_driver_static_swapout_node;
  driver->swapin      = plain_driver_static_swapin_node;
  driver->free_driver = plain_driver_static_free;
  {
    char *path;
    if (root_path != NULL)
      path = util_alloc_full_path(root_path , driver_path);
    else
      path = util_alloc_string_copy(driver_path);
    
    driver->path        = path_fmt_alloc_directory_fmt(path , true);
    free(path);
  }
  driver->plain_driver_static_id = PLAIN_DRIVER_STATIC_ID;
  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}
