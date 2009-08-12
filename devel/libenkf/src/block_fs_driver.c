#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <util.h>
#include <path_fmt.h>
#include <fs_types.h>
#include <buffer.h>
#include <basic_driver.h>
#include <block_fs_driver.h>
#include <block_fs.h>


struct block_fs_driver_struct {
  BASIC_DRIVER_FIELDS;
  int                block_fs_driver_id;
  fs_driver_type     driver_type;
  char             * mount_prefix;     /* ANALYZED | FORECAST | PARAMETER | STATIC : a string representation of the role this driver instance has. */
  block_fs_type   ** read_fs;
  block_fs_type   ** write_fs;
  
  char             * write_path;       /* root_path + the current write_case */
  char             * read_path;        /* root_path + the current read case  */
  char             * root_path;        /* The root path - as given by the ENSPATH directive. */

  /*-----------------------------------------------------------------*/
  /* Options which go directly to the underlying block_fs instances. */
  int              * block_size;
  bool             * internal_index;
  bool             * exernal_index;
  bool             * preload; 
};



static void block_fs_driver_assert_cast(block_fs_driver_type * block_fs_driver) {
  if (block_fs_driver->block_fs_driver_id != BLOCK_FS_DRIVER_ID) 
    util_abort("%s: internal error - cast failed - aborting \n",__func__);
}


static block_fs_driver_type * block_fs_driver_safe_cast( void * __driver) {
  block_fs_driver_type * driver = (block_fs_driver_type *) __driver;
  block_fs_driver_assert_cast(driver);
  return driver;
}


static char * block_fs_driver_alloc_key( const block_fs_driver_type * driver , const enkf_config_node_type * config_node , int report_step , int iens) {
  char * key = util_alloc_sprintf("%s.%d.%d" , enkf_config_node_get_key( config_node ) , report_step , iens);
  return key;
}


static block_fs_type * block_fs_driver_get_fs( block_fs_driver_type * driver , const enkf_config_node_type * config_node , bool read) {
  enkf_impl_type impl_type = enkf_config_node_get_impl_type( config_node );
  int fs_nr                = impl_type - IMPL_TYPE_OFFSET;
  block_fs_type * block_fs;
  
  if (read) 
    block_fs = driver->read_fs[fs_nr];
  else 
    block_fs = driver->write_fs[fs_nr];
  
  return block_fs;
}






static void block_fs_driver_load_node(void * _driver , const enkf_config_node_type * config_node , int report_step , int iens ,  buffer_type * buffer) {
  block_fs_driver_type * driver = block_fs_driver_safe_cast( _driver );
  {
    char * key               = block_fs_driver_alloc_key( driver , config_node , report_step , iens );
    block_fs_type * block_fs = block_fs_driver_get_fs( driver , config_node , true );
    
    block_fs_fread_realloc_buffer( block_fs , key , buffer);
    
    free( key );
  }
}




static void block_fs_driver_save_node(void * _driver , const enkf_config_node_type * config_node , int report_step , int iens ,  buffer_type * buffer) {
  block_fs_driver_type * driver = (block_fs_driver_type *) _driver;
  block_fs_driver_assert_cast(driver);
  {
    char * key               = block_fs_driver_alloc_key( driver , config_node , report_step , iens );
    block_fs_type * block_fs = block_fs_driver_get_fs( driver , config_node , false );
    
    block_fs_fwrite_buffer( block_fs , key , buffer);
    
    free( key );
  }
}


void block_fs_driver_unlink_node(void * _driver , const enkf_config_node_type * config_node, int report_step , int iens ) {
  block_fs_driver_type * driver = (block_fs_driver_type *) _driver;
  block_fs_driver_assert_cast(driver);
  {
    char * key               = block_fs_driver_alloc_key( driver , config_node , report_step , iens );
    block_fs_type * block_fs = block_fs_driver_get_fs( driver , config_node , true );
    block_fs_unlink_file( block_fs , key );
    free( key );
  }
}


/**
   Observe that the semantics is fundamentally different between
   block_fs_driver_paramater_has_node, and block_fs_driver_load_node:

   * When (trying to) load a node the function will try previous report steps
     all the way back to the first report step.

   * The has_node function will _not_ go back to earlier report steps, but
     instead return false if the report_step we ask for is not present.
*/

bool block_fs_driver_has_node(void * _driver , const enkf_config_node_type * config_node , int report_step , int iens ) {
  block_fs_driver_type * driver = (block_fs_driver_type *) _driver;
  block_fs_driver_assert_cast(driver);
  {
    char * key               = block_fs_driver_alloc_key( driver , config_node , report_step , iens );
    block_fs_type * block_fs = block_fs_driver_get_fs( driver , config_node , true );
    bool has_node            = block_fs_has_file( block_fs , key );
    free( key );
    return has_node;
  }
}






static void block_fs_mount_single_fs( block_fs_driver_type * driver , enkf_impl_type impl_type , bool read , int blocksize , int max_cache_size , bool internal_index , bool external_index , bool preload) {
  char * mount_file;
  char * base = util_alloc_sprintf("%s_%s" , driver->mount_prefix , enkf_types_get_impl_name( impl_type ));
  if (read) {
    mount_file = util_alloc_filename( driver->read_path , base , "mnt" );
    driver->read_fs[ impl_type - IMPL_TYPE_OFFSET ] = block_fs_mount( mount_file , blocksize , max_cache_size , internal_index, external_index , preload );
    free( mount_file );
  } else {
    mount_file = util_alloc_filename( driver->write_path , base , "mnt" );
    driver->write_fs[ impl_type - IMPL_TYPE_OFFSET ] = block_fs_mount( mount_file , blocksize , max_cache_size , internal_index, external_index , preload );
    free( mount_file );
  }
  free( base );
}



static void block_fs_create_new_fs( block_fs_driver_type * driver , bool read , bool mount ) {
  bool internal_index = true;
  bool external_index = true;
  const int STATIC_blocksize       = 1;
  const int MULTFLT_blocksize      = 1;
  const int FIELD_blocksize        = 1;
  const int GEN_KW_blocksize       = 1;
  const int HAVANA_FAULT_blocksize = 1;
  const int SUMMARY_blocksize      = 1;
  const int GEN_DATA_blocksize     = 1;
  const int max_cache_size         = 256; /* ~ 16 doubles */

  if (read) {
    driver->read_fs = util_malloc( (MAX_IMPL_TYPE - IMPL_TYPE_OFFSET + 1) * sizeof * driver->read_fs , __func__);
    for (int impl_type = IMPL_TYPE_OFFSET; impl_type <= MAX_IMPL_TYPE; impl_type++)
      driver->read_fs[impl_type - IMPL_TYPE_OFFSET] = NULL;
  } else {
    driver->write_fs = util_malloc( (MAX_IMPL_TYPE - IMPL_TYPE_OFFSET + 1) * sizeof * driver->write_fs , __func__);
    for (int impl_type = IMPL_TYPE_OFFSET; impl_type <= MAX_IMPL_TYPE; impl_type++)
      driver->write_fs[impl_type - IMPL_TYPE_OFFSET] = NULL;
  }

  if (mount) {
    switch( driver->driver_type) {
    case(DRIVER_STATIC):
      block_fs_mount_single_fs( driver , STATIC       , read , STATIC_blocksize        , max_cache_size , internal_index , external_index , false);
      break;
    case(DRIVER_PARAMETER):
      block_fs_mount_single_fs( driver , MULTFLT      , read , MULTFLT_blocksize       , max_cache_size , internal_index , external_index , false);
      block_fs_mount_single_fs( driver , FIELD        , read , FIELD_blocksize         , max_cache_size , internal_index , external_index , false);
      block_fs_mount_single_fs( driver , GEN_KW       , read , GEN_KW_blocksize        , max_cache_size , internal_index , external_index , true);
      block_fs_mount_single_fs( driver , HAVANA_FAULT , read , HAVANA_FAULT_blocksize  , max_cache_size , internal_index , external_index , false);
      block_fs_mount_single_fs( driver , GEN_DATA     , read , GEN_DATA_blocksize      , max_cache_size , internal_index , external_index , false);
      break;
    case(DRIVER_DYNAMIC_FORECAST):
      block_fs_mount_single_fs( driver , SUMMARY      , read , SUMMARY_blocksize       , max_cache_size , internal_index , external_index , true);
      block_fs_mount_single_fs( driver , FIELD        , read , FIELD_blocksize         , max_cache_size , internal_index , external_index , false);
      block_fs_mount_single_fs( driver , GEN_DATA     , read , GEN_DATA_blocksize      , max_cache_size , internal_index , external_index , false);
      break;
    case(DRIVER_DYNAMIC_ANALYZED):
      block_fs_mount_single_fs( driver , SUMMARY      , read , SUMMARY_blocksize       , max_cache_size , internal_index , external_index , true);
      block_fs_mount_single_fs( driver , FIELD        , read , FIELD_blocksize         , max_cache_size , internal_index , external_index , false);
      block_fs_mount_single_fs( driver , GEN_DATA     , read , GEN_DATA_blocksize      , max_cache_size , internal_index , external_index , false);
      break;
    default:
      util_abort("%s: driver_type:%d not recognized \n",__func__);
    }
  }
}



static void block_fs_close_fs( block_fs_driver_type * driver , bool read ) {
  int impl_type;
  if (read) {
    if (driver->read_fs != NULL) {
      for (impl_type = IMPL_TYPE_OFFSET; impl_type <= MAX_IMPL_TYPE; impl_type++) {
        block_fs_type * block_fs = driver->read_fs[ impl_type - IMPL_TYPE_OFFSET ];
        if (block_fs != NULL)  /* There are holes with NULL */
          block_fs_close( block_fs , true);
      }
      free( driver->read_fs );
    }
  } else {
    if (driver->write_fs != NULL) {
      for (impl_type = IMPL_TYPE_OFFSET; impl_type <= MAX_IMPL_TYPE; impl_type++) {
        block_fs_type * block_fs = driver->write_fs[ impl_type - IMPL_TYPE_OFFSET ];
        if (block_fs != NULL)
          block_fs_close( block_fs , true);
      }
      free( driver->write_fs );
    }
  }
}




/**
   The enkf_fs layer has already made certain that this directory is different
   from the current. 
*/
void block_fs_driver_select_dir(void *_driver , const char * directory, bool read) {
  block_fs_driver_type * driver = block_fs_driver_safe_cast(_driver);
  if (read) {
    if (util_string_equal( driver->write_path , driver->read_path)) {
      /* 
         Read and write have accessed the same fs, must create new pointers
         for the read_fs instances.
      */
      driver->read_path = util_realloc_filename( driver->read_path , driver->root_path , directory , NULL);
      util_make_path( driver->read_path );
      block_fs_create_new_fs( driver ,  true , true);
    } else {
      /* 
         When the read_fs is changed over, the underlying fs will go out of scope.
         Must close the current read_fs properly before continuing.
      */
      block_fs_close_fs( driver , true );
      driver->read_path = util_realloc_filename( driver->read_path , driver->root_path , directory , NULL);
      util_make_path( driver->read_path );
      if (util_string_equal( driver->read_path , driver->write_path)) {
        /* 
           The new read driver is equal the existing write_driver. We
           do not mount anything more, we just assign the read
           pointers to point to write pointers.
        */
        int impl_type;
        block_fs_create_new_fs( driver , true , false);
        for (impl_type = IMPL_TYPE_OFFSET; impl_type <= MAX_IMPL_TYPE; impl_type++) 
          driver->read_fs[impl_type - IMPL_TYPE_OFFSET] = driver->write_fs[impl_type - IMPL_TYPE_OFFSET];
        
      } else 
        /* 
           The new read driver is different from the existing write driver - we create a brand new; fully mounted fs.
        */
        block_fs_create_new_fs( driver , true , true);
    }
  } else {
    /*****************************************************************/
    /* We are selecting a new write driver. */
    if (util_string_equal( driver->write_path , driver->read_path)) {
      /* 
         Read and write have accessed the same fs, must create new pointers
         for the write_fs instances.
      */
      driver->write_path = util_realloc_filename( driver->write_path , driver->root_path , directory , NULL);
      util_make_path( driver->write_path );
      block_fs_create_new_fs( driver ,  false , true );
    } else {
      /* 
         When the write_fs is changed over, the underlying fs will go out of scope.
         Must close the current write_fs properly before continuing.
      */
      block_fs_close_fs( driver , false );
      driver->write_path = util_realloc_filename( driver->write_path , driver->root_path , directory , NULL);
      util_make_path( driver->write_path );
      if (util_string_equal( driver->read_path , driver->write_path)) {
        /* 
           The new write driver is equal the existing read_driver. We
           do not mount anything more, we just assign the write
           pointers to point to read pointers.
        */
        int impl_type;
        block_fs_create_new_fs( driver , false , false);
        for (impl_type = IMPL_TYPE_OFFSET; impl_type <= MAX_IMPL_TYPE; impl_type++) 
          driver->write_fs[impl_type - IMPL_TYPE_OFFSET] = driver->read_fs[impl_type - IMPL_TYPE_OFFSET];
        
      } else 
        /* 
           The new read driver is different from the existing write driver - we create a brand new; fully mounted fs.
        */
        block_fs_create_new_fs( driver , false , true);
    }
  }
}




void block_fs_driver_free(void *_driver) {
  block_fs_driver_type * driver = (block_fs_driver_type *) _driver;
  block_fs_driver_assert_cast(driver);

  block_fs_close_fs( driver , true );
  
  if (util_string_equal( driver->read_path , driver->write_path)) 
    /* Write has already been closed .*/
    free( driver->write_fs );
  else
    block_fs_close_fs( driver , false );

  free( driver->read_path );
  free( driver->write_path );
  util_safe_free( driver->root_path );
  util_safe_free( driver->mount_prefix );
  free(driver);
}




/**
  The driver takes a copy of the path object, i.e. it can be deleted
  in the calling scope after calling block_fs_driver_alloc().

  This is where the various function pointers are initialized.
*/
static void * block_fs_driver_alloc(const char * root_path , fs_driver_type driver_type) {
  block_fs_driver_type * driver = util_malloc(sizeof * driver , __func__);

  driver->load        	= block_fs_driver_load_node;
  driver->save        	= block_fs_driver_save_node;
  driver->free_driver 	= block_fs_driver_free;
  driver->unlink_node 	= block_fs_driver_unlink_node;
  driver->has_node    	= block_fs_driver_has_node;
  driver->select_dir    = block_fs_driver_select_dir;

  driver->write_fs      = NULL;
  driver->read_fs       = NULL;

  driver->root_path     = util_alloc_string_copy( root_path );
  driver->read_path     = NULL;
  driver->write_path    = NULL;
  driver->block_fs_driver_id = BLOCK_FS_DRIVER_ID;
  driver->driver_type        = driver_type;

  switch (driver_type) {
  case(DRIVER_PARAMETER):
    driver->mount_prefix = util_alloc_string_copy("PARAMETER");
    break;
  case(DRIVER_STATIC):
    driver->mount_prefix = util_alloc_string_copy("STATIC");
    break;
  case(DRIVER_DYNAMIC_FORECAST):
    driver->mount_prefix = util_alloc_string_copy("FORECAST");
    break;
  case(DRIVER_DYNAMIC_ANALYZED):
    driver->mount_prefix = util_alloc_string_copy("ANALYZED");
    break;
  default:
    util_abort("%s: driver_id:%d not recognized \n",__func__ , driver_type);
  }

  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}




void block_fs_driver_fwrite_mount_info(FILE * stream , fs_driver_type driver_type) {
  util_fwrite_int(driver_type , stream);
  util_fwrite_int(BLOCK_FS_DRIVER_ID , stream);
  util_fwrite_int(driver_type , stream );
}



/**
   The two integers from the mount info have already been read at the enkf_fs level.
*/

block_fs_driver_type * block_fs_driver_fread_alloc(const char * root_path , FILE * stream) {
  fs_driver_type driver_type = util_fread_int( stream );
  block_fs_driver_type * driver = block_fs_driver_alloc(root_path , driver_type );
  return driver;
}


