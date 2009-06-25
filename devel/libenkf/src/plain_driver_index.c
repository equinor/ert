#include <stdlib.h>
#include <path_fmt.h>
#include <util.h>
#include <fs_types.h>
#include <stringlist.h>
#include <plain_driver_common.h>
#include <plain_driver_index.h>
#include <basic_driver.h>


/*
  This file implements a plain index driver - it currently only stores
  the restart_kw_list - i.e. the ordering of keywords in ECLIPSE
  restart files.
*/

struct plain_driver_index_struct {
  BASIC_INDEX_DRIVER_FIELDS;
  int         __id;     
  path_fmt_type  * path_fmt;

  /* ---------------------------: The different parts of the path variable is documented in plain_driver_dynamic. */  
  char           * root_path;
  char           * fmt_string;
};


static plain_driver_index_type * plain_driver_index_safe_cast(void * __index_driver){
  plain_driver_index_type * index_driver = (plain_driver_index_type *) __index_driver;
  if (index_driver->__id != PLAIN_DRIVER_INDEX_ID)
    util_abort("%s: runtime cast failed \n",__func__);
  
  return index_driver;
}
  
void plain_driver_index_select_dir(void *_driver , const char * directory) {
  plain_driver_index_type * driver = plain_driver_index_safe_cast(_driver);
  driver->path_fmt = plain_driver_common_realloc_path_fmt(driver->path_fmt , driver->root_path , directory , driver->fmt_string);
}




void plain_driver_index_fwrite_restart_kw_list(void * __index_driver, int report_step , int iens , const stringlist_type * kw_list) {
  plain_driver_index_type * index_driver = plain_driver_index_safe_cast(__index_driver);
  {
    char * kw_file = path_fmt_alloc_file(index_driver->path_fmt , true , report_step , iens , "kw_list");
    FILE * stream  = util_fopen(kw_file , "w");
    stringlist_fwrite(kw_list , stream);
    fclose(stream);
    free(kw_file);
  }
}


void plain_driver_index_fread_restart_kw_list(void * __index_driver, int report_step, int iens , stringlist_type * kw_list) {
  plain_driver_index_type * index_driver = plain_driver_index_safe_cast(__index_driver);
  {
    char * kw_file = path_fmt_alloc_file(index_driver->path_fmt , false , report_step , iens , "kw_list");
    FILE * stream  = util_fopen(kw_file , "r");
    stringlist_fread(kw_list , stream);
    fclose(stream);
    free(kw_file);
  }
}


void plain_driver_index_free(void * __index_driver) {
  plain_driver_index_type * index_driver = plain_driver_index_safe_cast(__index_driver);
  {
    path_fmt_free( index_driver->path_fmt );
    util_safe_free( index_driver->root_path );
    util_safe_free( index_driver->fmt_string );
    free(index_driver);
    index_driver = NULL;
  }
}



void * plain_driver_index_alloc(const char * root_path , const char * index_path) {
  plain_driver_index_type * plain_driver = util_malloc(sizeof * plain_driver , __func__);
  plain_driver->select_dir  = plain_driver_index_select_dir;
  plain_driver->save_kwlist = plain_driver_index_fwrite_restart_kw_list;
  plain_driver->load_kwlist = plain_driver_index_fread_restart_kw_list;
  plain_driver->free_driver = plain_driver_index_free;

  plain_driver->root_path  = util_alloc_string_copy( root_path );
  plain_driver->fmt_string = util_alloc_string_copy( index_path );  
  plain_driver->path_fmt   = NULL;   
  
  plain_driver->__id = PLAIN_DRIVER_INDEX_ID;
  {
    basic_driver_index_type * basic_driver = (basic_driver_index_type *) plain_driver;
    basic_driver_index_init(basic_driver);
    return basic_driver;
  }
}



void * plain_driver_index_fread_alloc(const char * root_path , FILE * stream) {
  char * index_path        = util_fread_alloc_string( stream );
  plain_driver_index_type * plain_driver_index = plain_driver_index_alloc(root_path , index_path);

  free(index_path);
  return plain_driver_index;
}


void plain_driver_index_fwrite_mount_info(FILE * stream , bool read , const char * fmt) {
  util_fwrite_bool(read                  , stream); 
  util_fwrite_int(DRIVER_INDEX          , stream);
  util_fwrite_int(PLAIN_DRIVER_INDEX_ID , stream);
  util_fwrite_string(fmt , stream);
}

 
