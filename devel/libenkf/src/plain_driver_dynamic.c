#include <stdlib.h>
#include <enkf_node.h>
#include <basic_driver.h>
#include <plain_driver_dynamic.h>
#include <plain_driver_common.h>
#include <fs_types.h>
#include <path_fmt.h>
#include <util.h>



struct plain_driver_dynamic_struct {
  BASIC_DRIVER_FIELDS;
  int             __id;
  path_fmt_type * forecast_path;
  path_fmt_type * analyzed_path; 
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




void plain_driver_dynamic_free(void *_driver) {
  plain_driver_dynamic_type * driver = plain_driver_dynamic_safe_cast(_driver);
  path_fmt_free(driver->forecast_path);
  path_fmt_free(driver->analyzed_path);
  free(driver);
}



void plain_driver_dynamic_README(const char * root_path) {
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
  in the calling scope after calling plain_driver_dynamic_alloc().
*/
void * plain_driver_dynamic_alloc(const char * root_path , const char * forecast_path , const char * analyzed_path) {
  plain_driver_dynamic_type * driver = util_malloc(sizeof * driver , __func__);
  driver->load        	= plain_driver_dynamic_load_node;
  driver->save        	= plain_driver_dynamic_save_node;
  driver->has_node    	= plain_driver_dynamic_has_node;
  driver->free_driver 	= plain_driver_dynamic_free;
  driver->unlink_node 	= plain_driver_dynamic_unlink_node;
  driver->load_ensemble = NULL;
  driver->load_ts       = NULL;
  driver->save_ensemble = NULL;
  driver->save_ts       = NULL;
  {
    char *f_path;
    char *a_path;
    if (root_path != NULL) {
      f_path = util_alloc_full_path(root_path , forecast_path);
      a_path = util_alloc_full_path(root_path , analyzed_path);
    } else {
      f_path = util_alloc_string_copy(forecast_path);
      a_path = util_alloc_string_copy(analyzed_path);
    }
    
    driver->forecast_path = path_fmt_alloc_directory_fmt( f_path );
    driver->analyzed_path = path_fmt_alloc_directory_fmt( a_path );
    free(a_path);
    free(f_path);
  }
  driver->__id = PLAIN_DRIVER_DYNAMIC_ID;
  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}


void plain_driver_dynamic_fwrite_mount_info(FILE * stream , const char * forecast_fmt , const char * analyzed_fmt) {
  util_fwrite_int(DYNAMIC_DRIVER , stream);
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

