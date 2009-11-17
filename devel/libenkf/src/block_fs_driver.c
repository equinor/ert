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
#include <enkf_types.h>
#include <thread_pool.h>

typedef struct bfs_struct bfs_type;

struct bfs_struct  {
  block_fs_type * read_fs;
  block_fs_type * write_fs;

  char          * read_path;
  char          * write_path;

  /*****************************************************************/
  /* All fields below here are info which go straight to the block_fs system. */ 
  char          * mount_basefile;
  bool            preload;
  int             block_size;
  int             max_cache_size;
};



struct block_fs_driver_struct {
  BASIC_DRIVER_FIELDS;
  int                __id;
  int                num_drivers;
  fs_driver_type     driver_type;
  block_fs_type   ** read_fs;
  block_fs_type   ** write_fs;
  bfs_type        ** fs_list;
  
  char             * root_path;        /* The root path - as given by the ENSPATH directive. */
}; 



static void bfs_close( bfs_type * bfs ) {
  bool equal = ( bfs->read_fs == bfs->write_fs );
  if (bfs->read_fs != NULL)
    block_fs_close( bfs->read_fs , true);
  if (!equal)
    block_fs_close( bfs->write_fs , true);

  util_safe_free( bfs->read_path );
  util_safe_free( bfs->write_path );
  free( bfs->mount_basefile );
  free( bfs );
}


static void * bfs_close__( void * arg ) {
  bfs_type * bfs = ( bfs_type * ) arg;
  bfs_close( bfs );
  
  return NULL;
}


static bfs_type * bfs_alloc( fs_driver_type driver_type , int block_size , int max_cache_size , bool preload) {
  bfs_type * fs      = util_malloc( sizeof * fs , __func__);
  fs->read_fs        = NULL;
  fs->write_fs       = NULL;
  fs->read_path      = NULL;
  fs->write_path     = NULL;
  fs->block_size     = block_size;
  fs->max_cache_size = max_cache_size;
  fs->preload        = preload;
  fs->mount_basefile = util_alloc_sprintf("%s" , fs_types_get_driver_name( driver_type ) );
  return fs;
}




static bfs_type ** bfs_alloc_driver_list( int num_drivers , fs_driver_type driver_type , int block_size , int max_cache_size , bool preload) {
  int i;
  bfs_type ** driver_list = util_malloc( num_drivers * sizeof * driver_list , __func__); 
  for (i=0; i < num_drivers; i++)
    driver_list[i] = bfs_alloc( driver_type , block_size , max_cache_size , preload);
  return driver_list;
}



/**
   The enkf_fs layer has already made certain that this directory is
   different from the current.
*/

static void bfs_select_dir( bfs_type * bfs , const char * root_path , const char * directory , bool read ) {
  const int fsync_interval      =  10;     /* An fsync() call is issued for every 10'th write. */
  const int fragmentation_limit = 1.0;     /* 1.0 => NO defrag is run. */
  if (read) {
    if (bfs->read_fs == bfs->write_fs) {
      /**
         Read an write currently mount the same fs. Must open a
         new fs instance for the reading. 
      */

      bfs->read_path = util_realloc_filename( bfs->read_path , root_path , directory , NULL);
      util_make_path( bfs->read_path );
      {
        char * mount_file = util_alloc_filename( bfs->read_path , bfs->mount_basefile , "mnt" );
        bfs->read_fs      = block_fs_mount( mount_file , bfs->block_size , bfs->max_cache_size , fragmentation_limit , fsync_interval , bfs->preload , false );
        free( mount_file );
      }
    } else {
      /* 
         They are currently different:
         
         1. Must close the existing read fs properly.
         2. Read/write Might become equal after the select?? 
      */
      if (bfs->read_fs != NULL) block_fs_close( bfs->read_fs , true);
      bfs->read_path = util_realloc_filename( bfs->read_path , root_path , directory , NULL);
      if (util_string_equal( bfs->read_path , bfs->write_path))
        bfs->read_fs = bfs->write_fs;  /* OK - the same underlying storage. */
      else {
        char * mount_file = util_alloc_filename( bfs->read_path , bfs->mount_basefile , "mnt" );
        util_make_path( bfs->read_path );
        bfs->read_fs      = block_fs_mount( mount_file , bfs->block_size , bfs->max_cache_size , fragmentation_limit , fsync_interval , bfs->preload , false );
        free( mount_file );
      }
    }
  } else {
    if (bfs->read_fs == bfs->write_fs) {
      /**
         Read an write currently mount the same fs. Must open a
         new fs instance for the writing. 
      */

      bfs->write_path = util_realloc_filename( bfs->write_path , root_path , directory , NULL);
      util_make_path( bfs->write_path );
      {
        char * mount_file = util_alloc_filename( bfs->write_path , bfs->mount_basefile , "mnt" );
        bfs->write_fs      = block_fs_mount( mount_file , bfs->block_size , bfs->max_cache_size , fragmentation_limit , fsync_interval , bfs->preload , false );
        free( mount_file );
      }
    } else {
      /* They are currently different. Might become equal after the select?? */
      if (bfs->write_fs != NULL) block_fs_close( bfs->write_fs , true);
      bfs->write_path = util_realloc_filename( bfs->write_path , root_path , directory , NULL);
      if (util_string_equal( bfs->read_path , bfs->write_path))
        bfs->write_fs = bfs->read_fs;  /* OK - the same underlying storage. */
      else {
        char * mount_file = util_alloc_filename( bfs->write_path , bfs->mount_basefile , "mnt" );
        util_make_path( bfs->write_path );
        bfs->write_fs     = block_fs_mount( mount_file , bfs->block_size , bfs->max_cache_size , fragmentation_limit , fsync_interval , bfs->preload , false );
        free( mount_file );
      }
    }
  }
}



static void * bfs_select_dir__(void * arg) {
  arg_pack_type * arg_pack = arg_pack_safe_cast( arg );
  bfs_type * bfs         = arg_pack_iget_ptr( arg_pack , 0 );
  const char * root_path = arg_pack_iget_ptr( arg_pack , 1 );
  const char * directory = arg_pack_iget_ptr( arg_pack , 2 );
  bool  read             = arg_pack_iget_bool( arg_pack , 3 );

  bfs_select_dir( bfs , root_path , directory , read );
  return NULL;
}


static void bfs_fsync( bfs_type * bfs ) {
  block_fs_fsync( bfs->read_fs );
  if (bfs->write_fs != bfs->read_fs)
    block_fs_fsync( bfs->write_fs );
}



/*****************************************************************/


static void block_fs_driver_assert_cast(block_fs_driver_type * block_fs_driver) {
  if (block_fs_driver->__id != BLOCK_FS_DRIVER_ID) 
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


/**
   This function will take an input string, and try to to parse it as
   string.int.int, where string is the normal enkf key, and the two
   integers are report_step and ensemble number respectively. The
   storage for the enkf_key is allocated here in this function, and
   must be freed by the calling scope.  

   If the parsing fails the function will return false, and *config_key
   will be set to NULL; in this case the report_step and iens poinyers
   will not be touched.
*/

bool block_fs_sscanf_key(const char * key , char ** config_key , int * __report_step , int * __iens) {
  char ** tmp;
  int num_items;

  *config_key = NULL;
  util_split_string(key , "." , &num_items , &tmp);  /* The key can contain additional '.' - can not use sscanf(). */
  if (num_items >= 3) {
    int report_step , iens;
    if (util_sscanf_int(tmp[num_items - 2] , &report_step) && util_sscanf_int(tmp[num_items - 1] , &iens)) {
      /* OK - all is hunkadory */
      *__report_step = report_step;
      *__iens        = iens;
      *config_key    = util_alloc_joined_string((const char **) tmp , num_items - 2 , ".");  /* This must bee freed by the calling scope */
      util_free_stringlist( tmp , num_items );
      return true;
    } else  
      /* Failed to parse the two last items as integers. */
      return false;
  } else
    /* Did not have at least three items. */
    return false;
}




static bfs_type * block_fs_driver_get_fs( block_fs_driver_type * driver , int iens ) {
  int phase                = (iens % driver->num_drivers);
  
  return driver->fs_list[phase];
}




static void block_fs_driver_load_node(void * _driver , const enkf_config_node_type * config_node , int report_step , int iens ,  buffer_type * buffer) {
  block_fs_driver_type * driver = block_fs_driver_safe_cast( _driver );
  {
    char * key          = block_fs_driver_alloc_key( driver , config_node , report_step , iens );
    bfs_type      * bfs = block_fs_driver_get_fs( driver , iens );
    
    block_fs_fread_realloc_buffer( bfs->read_fs , key , buffer);
    
    free( key );
  }
}




static void block_fs_driver_save_node(void * _driver , const enkf_config_node_type * config_node , int report_step , int iens ,  buffer_type * buffer) {
  block_fs_driver_type * driver = (block_fs_driver_type *) _driver;
  block_fs_driver_assert_cast(driver);
  {
    char * key     = block_fs_driver_alloc_key( driver , config_node , report_step , iens );
    bfs_type * bfs = block_fs_driver_get_fs( driver , iens );
    block_fs_fwrite_buffer( bfs->write_fs , key , buffer);
    
    free( key );
  }
}


void block_fs_driver_unlink_node(void * _driver , const enkf_config_node_type * config_node, int report_step , int iens ) {
  block_fs_driver_type * driver = (block_fs_driver_type *) _driver;
  block_fs_driver_assert_cast(driver);
  {
    char * key     = block_fs_driver_alloc_key( driver , config_node , report_step , iens );
    bfs_type * bfs = block_fs_driver_get_fs( driver , iens );
    block_fs_unlink_file( bfs->read_fs , key );
    free( key );
  }
}


bool block_fs_driver_has_node(void * _driver , const enkf_config_node_type * config_node , int report_step , int iens ) {
  block_fs_driver_type * driver = (block_fs_driver_type *) _driver;
  block_fs_driver_assert_cast(driver);
  {
    char * key      = block_fs_driver_alloc_key( driver , config_node , report_step , iens );
    bfs_type  * bfs = block_fs_driver_get_fs( driver , iens );
    bool has_node   = block_fs_has_file( bfs->read_fs , key );
    free( key );
    return has_node;
  }
}






/**
   The enkf_fs layer has already made certain that this directory is different
   from the current. 
*/

void block_fs_driver_select_dir(void *_driver , const char * directory, bool read) {
  block_fs_driver_type * driver = block_fs_driver_safe_cast(_driver);
  thread_pool_type * tp         = thread_pool_alloc( driver->num_drivers ); /* Maaany threads .... */
  arg_pack_type ** arglist      = util_malloc( sizeof * arglist * driver->num_drivers , __func__);
  msg_type * msg = msg_alloc("Mounting: ");
  int driver_nr;
  
  msg_show( msg );
  for (driver_nr = 0; driver_nr < driver->num_drivers; driver_nr++) {
    char * path = util_alloc_sprintf("%s%cmod_%d" , directory , UTIL_PATH_SEP_CHAR , driver_nr);
    arglist[driver_nr] = arg_pack_alloc();

    arg_pack_append_ptr( arglist[driver_nr] , driver->fs_list[driver_nr] );
    arg_pack_append_ptr( arglist[driver_nr] , driver->root_path );
    arg_pack_append_owned_ptr( arglist[driver_nr] , path , free);
    arg_pack_append_bool( arglist[driver_nr] , read );

    msg_update( msg , path );
    thread_pool_add_job( tp , bfs_select_dir__ , arglist[driver_nr] );
  }
  thread_pool_join( tp );
  msg_free( msg , true );

  for (driver_nr = 0; driver_nr < driver->num_drivers; driver_nr++) 
    arg_pack_free( arglist[driver_nr] );
  free( arglist );
  thread_pool_free( tp );
}






void block_fs_driver_free(void *_driver) {
  block_fs_driver_type * driver = block_fs_driver_safe_cast( _driver );
  {
    int driver_nr;
    thread_pool_type * tp         = thread_pool_alloc( driver->num_drivers );
    for (driver_nr = 0; driver_nr < driver->num_drivers; driver_nr++) 
      thread_pool_add_job( tp , bfs_close__ , driver->fs_list[driver_nr] );

    thread_pool_join( tp );
    thread_pool_free( tp );
  }
  free( driver->fs_list );
  util_safe_free( driver->root_path );
  free(driver);
}



static void block_fs_driver_fsync( void * _driver ) {
  block_fs_driver_type * driver = (block_fs_driver_type *) _driver;
  block_fs_driver_assert_cast(driver);
  
  {
    int driver_nr;
    block_fs_driver_type * driver = block_fs_driver_safe_cast(_driver);
    for (driver_nr = 0; driver_nr < driver->num_drivers; driver_nr++)  
      bfs_fsync( driver->fs_list[driver_nr] );
  }
}



/**
  The driver takes a copy of the path object, i.e. it can be deleted
  in the calling scope after calling block_fs_driver_alloc().

  This is where the various function pointers are initialized.
*/
static void * block_fs_driver_alloc(const char * root_path , fs_driver_type driver_type , int num_drivers ) {
  block_fs_driver_type * driver = util_malloc(sizeof * driver , __func__);

  driver->load        	= block_fs_driver_load_node;
  driver->save        	= block_fs_driver_save_node;
  driver->free_driver 	= block_fs_driver_free;
  driver->unlink_node 	= block_fs_driver_unlink_node;
  driver->has_node    	= block_fs_driver_has_node;
  driver->select_dir    = block_fs_driver_select_dir;
  driver->fsync_driver  = block_fs_driver_fsync;

  driver->write_fs      = NULL;
  driver->read_fs       = NULL;
  driver->root_path     = util_alloc_string_copy( root_path );
  driver->__id          = BLOCK_FS_DRIVER_ID;
  driver->driver_type   = driver_type;
  driver->num_drivers   = num_drivers;

  {
    const int STATIC_blocksize       = 64;
    const int PARAMETER_blocksize    = 64;
    const int DYNAMIC_blocksize      = 64;
    const int max_cache_size         = 512; /* ~ 32 doubles */
  
    switch( driver->driver_type) {
    case(DRIVER_STATIC):
      driver->fs_list = bfs_alloc_driver_list( driver->num_drivers , DRIVER_STATIC , STATIC_blocksize , max_cache_size , false);
      break;
    case(DRIVER_PARAMETER):
      driver->fs_list  = bfs_alloc_driver_list( driver->num_drivers , DRIVER_PARAMETER , PARAMETER_blocksize    , max_cache_size , false);
      break;
    case(DRIVER_DYNAMIC_FORECAST):
      driver->fs_list = bfs_alloc_driver_list( driver->num_drivers , DRIVER_DYNAMIC_FORECAST , DYNAMIC_blocksize , max_cache_size , true);
      break;
    case(DRIVER_DYNAMIC_ANALYZED):
      driver->fs_list = bfs_alloc_driver_list( driver->num_drivers , DRIVER_DYNAMIC_ANALYZED , DYNAMIC_blocksize , max_cache_size , true);
      break;
    default:
      util_abort("%s: driver_type:%d not recognized \n",__func__);
    }
  }
  
  
  {
    basic_driver_type * basic_driver = (basic_driver_type *) driver;
    basic_driver_init(basic_driver);
    return basic_driver;
  }
}




void block_fs_driver_fwrite_mount_info(FILE * stream , fs_driver_type driver_type , int num_drivers ) {
  util_fwrite_int(driver_type        , stream);
  util_fwrite_int(BLOCK_FS_DRIVER_ID , stream);
  util_fwrite_int(driver_type        , stream );
  util_fwrite_int(num_drivers        , stream );
}



/**
   The two integers from the mount info have already been read at the enkf_fs level.
*/

block_fs_driver_type * block_fs_driver_fread_alloc(const char * root_path , FILE * stream) {
  fs_driver_type driver_type = util_fread_int( stream );
  int num_drivers            = util_fread_int( stream );
  block_fs_driver_type * driver = block_fs_driver_alloc(root_path , driver_type , num_drivers );
  return driver;
}


