#include <stdlib.h>
#include <path_fmt.h>
#include <util.h>
#include <fs_types.h>
#include <stringlist.h>
#include <block_fs.h>
#include <block_fs_driver_index.h>
#include <basic_driver.h>


/*
  this file implements a plain index driver - it currently only stores
  the restart_kw_list - i.e. the ordering of keywords in eclipse
  restart files.
*/

struct block_fs_driver_index_struct {
  BASIC_INDEX_DRIVER_FIELDS;
  int                __id;
  block_fs_type    * read_fs;
  block_fs_type    * write_fs;
  
  char             * write_path;       /* root_path + the current write_case */
  char             * read_path;        /* root_path + the current read case  */
  char             * root_path;        /* the root path - as given by the enspath directive. */

  /*-----------------------------------------------------------------*/
  /* options which go directly to the underlying block_fs instances. */
  int              block_size;
};


static block_fs_driver_index_type * block_fs_driver_index_safe_cast(void * __index_driver){
  block_fs_driver_index_type * index_driver = (block_fs_driver_index_type *) __index_driver;
  if (index_driver->__id != BLOCK_FS_DRIVER_INDEX_ID)
    util_abort("%s: runtime cast failed \n",__func__);
  
  return index_driver;
}

static void block_fs_driver_index_single_mount( block_fs_driver_index_type * driver , bool read) {
  const int  blocksize      = driver->block_size;
  char * mount_file;
  if (read) {
    mount_file = util_alloc_filename( driver->read_path , "INDEX" , "mnt");
    driver->read_fs = block_fs_mount( mount_file , blocksize , 0 , 0.0 , false , false);
  } else {
    mount_file = util_alloc_filename( driver->write_path , "INDEX" , "mnt");
    driver->write_fs = block_fs_mount( mount_file , blocksize , 0 , 0.0 , false , false);
  }
  free( mount_file );
}



  
void block_fs_driver_index_select_dir(void *_driver , const char * directory, bool read) {

  block_fs_driver_index_type * driver = block_fs_driver_index_safe_cast(_driver);

  if (read) {
    if (util_string_equal( driver->write_path , driver->read_path)) {
      /* 
         Read and write have accessed the same fs, must create new pointers
         for the read_fs instances.
      */
      driver->read_path = util_realloc_filename( driver->read_path , driver->root_path , directory , NULL);
      util_make_path( driver->read_path );
      block_fs_driver_index_single_mount( driver , true );
    } else {
      /*
        Close the existing read driver. 
      */
      if (driver->read_fs != NULL) block_fs_close( driver->read_fs , true );
      driver->read_path = util_realloc_filename( driver->read_path , driver->root_path , directory , NULL);
      util_make_path( driver->read_path );
      if (util_string_equal( driver->read_path , driver->write_path)) 
        driver->read_fs = driver->write_fs;
      else 
        block_fs_driver_index_single_mount( driver , true );
    }
  } else {
    if (util_string_equal( driver->write_path , driver->read_path)) {
      /* 
         Read and write have accessed the same fs, must create new pointers
         for the read_fs instances.
      */
      driver->write_path = util_realloc_filename( driver->write_path , driver->root_path , directory , NULL);
      util_make_path( driver->write_path );
      block_fs_driver_index_single_mount( driver , true );
    } else {
      /*
        Close the existing read driver. 
      */
      if (driver->write_fs != NULL) block_fs_close( driver->write_fs , true );
      driver->write_path = util_realloc_filename( driver->write_path , driver->root_path , directory , NULL);
      util_make_path( driver->write_path );
      if (util_string_equal( driver->read_path , driver->write_path)) 
        driver->write_fs = driver->read_fs;
      else 
        block_fs_driver_index_single_mount( driver , true );
    }
  }
}





void block_fs_driver_index_fwrite_restart_kw_list(void * __index_driver, int report_step , int iens , buffer_type * buffer) {
  block_fs_driver_index_type * index_driver = block_fs_driver_index_safe_cast(__index_driver);
  {
    char * key = util_alloc_sprintf("kw_list_%d.%d" , report_step , iens);
    block_fs_fwrite_buffer( index_driver->write_fs , key , buffer);
    free( key );
  }
}



void block_fs_driver_index_fread_restart_kw_list(void * __index_driver, int report_step, int iens , buffer_type * buffer) {
  block_fs_driver_index_type * index_driver = block_fs_driver_index_safe_cast(__index_driver);
  {
    char * key = util_alloc_sprintf("kw_list_%d.%d" , report_step , iens);
    block_fs_fread_realloc_buffer( index_driver->read_fs , key , buffer);
    free(key);
  }
}


void block_fs_driver_index_free(void * __index_driver) {
  block_fs_driver_index_type * driver = block_fs_driver_index_safe_cast(__index_driver);
  {
    block_fs_close( driver->read_fs , true);
    if (!util_string_equal( driver->read_path , driver->write_path)) 
      block_fs_close( driver->write_fs , true);
    
    free(driver);
    driver = NULL;
  }
}



void * block_fs_driver_index_alloc(const char * root_path) {
  block_fs_driver_index_type * block_fs_driver = util_malloc(sizeof * block_fs_driver , __func__);
  block_fs_driver->select_dir  = block_fs_driver_index_select_dir;
  block_fs_driver->save_kwlist = block_fs_driver_index_fwrite_restart_kw_list;
  block_fs_driver->load_kwlist = block_fs_driver_index_fread_restart_kw_list;
  block_fs_driver->free_driver = block_fs_driver_index_free;

  block_fs_driver->block_size     = 8;
  block_fs_driver->root_path      = util_alloc_string_copy( root_path );
  block_fs_driver->read_path      = NULL;
  block_fs_driver->write_path     = NULL;
  block_fs_driver->read_fs        = NULL;
  block_fs_driver->write_fs       = NULL;
  
  block_fs_driver->__id = BLOCK_FS_DRIVER_INDEX_ID;
  {
    basic_driver_index_type * basic_driver = (basic_driver_index_type *) block_fs_driver;
    basic_driver_index_init(basic_driver);
    return basic_driver;
  }
}



block_fs_driver_index_type * block_fs_driver_index_fread_alloc(const char * root_path , FILE * stream) {
  block_fs_driver_index_type * block_fs_driver_index = block_fs_driver_index_alloc( root_path );
  return block_fs_driver_index;
}


void block_fs_driver_index_fwrite_mount_info(FILE * stream ) {
  util_fwrite_int(DRIVER_INDEX             , stream);
  util_fwrite_int(BLOCK_FS_DRIVER_INDEX_ID , stream);
}

 
