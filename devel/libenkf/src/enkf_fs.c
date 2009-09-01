#include <sys/types.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <util.h>
#include <set.h>
#include <enkf_fs.h>
#include <enkf_defaults.h>
#include <msg.h>
#include <path_fmt.h>
#include <enkf_node.h>
#include <basic_driver.h>
#include <dirent.h>
#include <fs_types.h>
#include <menu.h>
#include <arg_pack.h>
#include <ecl_static_kw.h>
#include <plain_driver_index.h>
#include <plain_driver.h>
#include <sqlite3_driver.h>
#include <block_fs_driver.h>
#include <block_fs_driver_index.h>
#include <stringlist.h>
#include <arg_pack.h>
#include <pthread.h>

#define FS_MAGIC_ID         123998L
#define CURRENT_FS_VERSION  104



/**
   Version history:

   0  : 

   
   File system version            | First svn version     |  Last svn version
   --------------------------------------------------------------------------
   100                            |                       |  1799
   101                            |   1810                |  1886
   102                            |   1887/1902           |  1996 
   103                            |   1997                |
                                  |   2047: sqlite added  |  2125
   104                            |   2127                |
                                  |   2140: block_fs_index added
   --------------------------------------------------------------------------


   Version: 100
   ------------
   First version with subdirectories for different cases.


   
   Version: 101
   ------------
   Have changed the format for storing static keywords. Instead of a
   directory with numbered files in them, the keywords get an integer
   appended:

     INTEHEAD/0   ==>  INTEHEAD_0

   The actual changing of keyword is done in the function
   __realloc_static_kw() in enkf_state.c.



   Version: 102 
   ------------ 
   Have removed the type spesific plain drivers, now it is only
   plain_driver and plain_driver index. The special functionality for
   parameters/static/dynamic is now implemented at the enkf_fs level.
   The upgrade from 101 only involves the mount info and the
   implementation, the actual stored files are not touched by this
   upgrade.



   Version: 103
   ------------
   Have changed the reading/writing to go through the buffer
   type. This should simplify introducing other drivers than the
   simple plain file based driver.  

   The upgrade to version is 103 is quite extensive - all node types
   have specific _103_ functions. The xxx_fread() and xxx_fwrite()
   functions were removed in svn:2046.

   A very experimental version of the sqlite driver was added in
   svn:2047. When (if ??) this stabilizes it should probably lead to
   an upgrade to version 104.

   At the very last checkin of this version the fs prototypes also
   changed to use (const enkf_config_node_type *) instances instead of
   (const char * ) for the main key.


   
   Version: 104 
   ------------ 
   In this version the treatment of read_xx and write_xx drivers has
   changed. There are no longer separate read_xx and write_xx drivers,
   instead the drivers have internal state which differentiates
   between read and write.

*/



/**

  About storage in the EnKF system
  ================================

  The system for storage in the EnKF system is quite complicated, maybe too
  complicated. The reason the system is so complex is (at least) twofold:

    1. It is a goal that it should be relatively easy to write new
       systems (i.e. drivers) for storage. The current suite of
       drivers (plain_driver_xx) are based on normal (compressed)
       fread() and fwrite() system calls. But with the current
       implementation one could write e.g. a MySQL based driver for
       storage without touching(??) the rest of the EnKF code.

    2. The parameters / static restart data / dynamic restart data
       have very different storage characteristics. By splitting the
       storage up in different drivers we can write drivers which are
       specialized for the different types of data.


  

  The interface
  -------------     

  The unit of storage in the enkf_fs system is one enkf_node instance. The
  interface between the storage system and the rest of the EnKF system is
  through the enkf_fs functions:

    enkf_fs_fread_node()
    enkf_fs_has_node()
    enkf_fs_fwrite_node()
    enkf_fs_unlink_node()  - not implemented yet.


  So all these functions (partly except enkf_fs_has_node()) work on a enkf_node
  instance, and in addition they take the following input:

    - iens        : ensemble member number
    - report_step : the report_step number we are interested in
    - state       : whether we are considering an analyzed node or a forecast.  

  In addition to the functions enkf_fs_fread_node() and enkf_fs_fwrite_node() there
  are higher level functions enkf_fs_fread_alloc_ensemble() to load an ensemble of
  nodes and enkf_fs_fread_alloc_ts() to load a time series of nodes. The driver can
  implement these functions (to get a performance gain when loading many nodes). It
  is not necceasry to implement these functions, the enkf_fs layer has simple
  functions doing repeated calls to the enkf_fs_fread_node() function.


  The drivers
  -----------

  The enkf_fs layer does not self implement the functions to read and write
  nodes. Instead what happens is:

    1. We determine the type of the node (static/dynamic/parameter), and select
       the appropriate driver.

    2. The appropriate driver is called to implement e.g. the fread_node
       functions.

  The different types of data have different characteristcs, which the driver is
  implemented to support. The characteristics the drivers support are the
  following:


  dynamic driver
  --------------
  This is the simplest driver, all data is stored both in a forecast version and
  an analyzed version.


  parameter driver
  ----------------
  This driver utilizes that parameters do not change during the forward model,
  i.e. (analyzed , t) = (forecast , t - 1). So, only one version of the data is
  actually stored; if you ask for the forecast you just get the data from the
  previous report_step. 
  To support spin-ups and such the driver will actually go backwards in
  report_time all the way until a node is found on disk.


  static driver
  -------------
  Like the parameter driver this also only stores one version od the data,
  however in addition it has to query the node for a ID to support multiply
  occuring keywords in ECLIPSE restart files.

  Currently only the plain_driver_xxx family has been implemented. Observe that
  there is no dependencies between the drivers, it is perfectly possible to
  implement a new driver for storage of static datat only. (There is probably a
  large amount of static data which is common both between members and for
  several consecutive report steps; utilizing that one could write a static
  driver which was admittedly slower, but leaner on the storage.)

  The drivers are allocated prior to allocating the enkf_fs instance, and
  pointers are passed in when allocating the enkf_fs instance.


  Mounting the filesystem
  -----------------------

  Mounting the filesystem - cool ehh?? Anyway, the important point is
  that the moment ensemble information has hit the filesystem later
  versions of the enkf program must support exactly that lay-out,
  those drivers+++. To ensure this I see two possibilities:

    1. We can freeze the filesystem drivers, and the layout on disk
       indefinetly.

    2. We can store the information needed to bootstrap the drivers,
       according to the current layout on disk, in the
       filesystem. I.e. something like a '/etc/fstab' file.

  We have chosen the second alternative. Currently this implemented as
  follows:

    1. In main() we query for the file {root-path}/enkf_mount_info. If
       that file does not exists it is created by calls to the
       selected drivers xxxx_fwrite_mount_info() functions.

    2. enkf_fs_mount() is called with the enkf_mount_info as input.

  The enkf_mount_info file (BINARY) consists of four records (one for
  each driver, including the index). The format of each record is:

     DRIVER_CATEGORY   DRIVER_ID    INFO 
     int               int          void *

  The driver category should be one of the four integer values in
  fs_driver_type (fs_types.h) and DRIVER_ID is one the integer
  values in fs_driver_impl. The last void * data is whatever
  (serialized) info the driver needs to bootstrap. This info is
  written by the drivers xxxx_fwrite_mount_info() function, and it is
  used when the driver is allocated with xxxx_fread_alloc().

  The different drivers can be in arbitrary order in the
  enkf_mount_info file, but when foru records are read it checks that
  all drivers have been initialized, and aborts if that is not the
  case.
  
  If the enkf_mount_info file is deleted you (might) be fucked. It
  is currently 'protected' with chomd a-w - but that is of course not
  foolprof.
*/




/**
   Observe the following convention: the initial ensemble at report
   step 0 is supposed to be analyzed. If we ask for the forecast at
   report_step 0, we should get the analyzed value.
*/



struct enkf_fs_struct {
  
  basic_driver_type         * dynamic_forecast;
  basic_driver_type         * dynamic_analyzed;
  basic_driver_type  	    * parameter;
  basic_driver_type  	    * eclipse_static;
  basic_driver_index_type   * index ;

  bool                        use_locking;           /* */ 
  bool                        read_only;             /* Whether this filesystem has been mounted read-only. */
  int                         lock_fd;               /* Integer containing a file descriptor to lockfile. */
  char                      * lockfile;              /* Filename for lockfile. */

  set_type                  * dir_set;               /* Set containing the existing directories. */
  char                      * current_read_dir;      /* The currently active "directory" for reading. */
  char                      * current_write_dir;     /* The currently active directory fro writing. */ 
  char                      * mount_map;             /* Binary file containing all the information the filesystem needs to bootstrap itself. Updated when e.g. new directories are added. */
  long int                     __dir_offset;         /* The offset into the mount map where the directory information starts - very internal. */
};

/*****************************************************************/


/** 
    This function writes a new default mount map; to be called if no
    mount map is found. Returns the __dir_offset.
*/

long int enkf_fs_fwrite_new_mount_map(const char * mount_map, const char * default_dir, fs_driver_impl driver_impl) {
  long int  __dir_offset;
  FILE * stream = util_fopen( mount_map , "w");
  util_fwrite_long(FS_MAGIC_ID , stream);
  util_fwrite_int(CURRENT_FS_VERSION , stream);

  if (driver_impl == PLAIN_DRIVER_ID) {
    /* Writing the mount map for brand new PLAIN - fs. */
    plain_driver_fwrite_mount_info( stream , DRIVER_PARAMETER 	     , DEFAULT_PLAIN_PARAMETER_PATH);
    plain_driver_fwrite_mount_info( stream , DRIVER_STATIC    	     , DEFAULT_PLAIN_STATIC_PATH);
    plain_driver_fwrite_mount_info( stream , DRIVER_DYNAMIC_FORECAST , DEFAULT_PLAIN_DYNAMIC_FORECAST_PATH);
    plain_driver_fwrite_mount_info( stream , DRIVER_DYNAMIC_ANALYZED , DEFAULT_PLAIN_DYNAMIC_ANALYZED_PATH);
    plain_driver_index_fwrite_mount_info( stream ,                     DEFAULT_PLAIN_INDEX_PATH);
  } else if (driver_impl == SQLITE_DRIVER_ID) {
    /* Writing a new mount map for a SQLite based approach. */
    sqlite3_driver_fwrite_mount_info( stream , DRIVER_PARAMETER        , DEFAULT_SQLITE_PARAMETER_DBFILE);
    sqlite3_driver_fwrite_mount_info( stream , DRIVER_STATIC           , DEFAULT_SQLITE_STATIC_DBFILE);
    sqlite3_driver_fwrite_mount_info( stream , DRIVER_DYNAMIC_FORECAST , DEFAULT_SQLITE_DYNAMIC_FORECAST_DBFILE);
    sqlite3_driver_fwrite_mount_info( stream , DRIVER_DYNAMIC_ANALYZED , DEFAULT_SQLITE_DYNAMIC_ANALYZED_DBFILE);
    plain_driver_index_fwrite_mount_info( stream ,                       DEFAULT_PLAIN_INDEX_PATH);   /* Using the plain index driver. */
  } else if (driver_impl == BLOCK_FS_DRIVER_ID) {
    block_fs_driver_fwrite_mount_info( stream , DRIVER_PARAMETER 	);
    block_fs_driver_fwrite_mount_info( stream , DRIVER_STATIC    	);
    block_fs_driver_fwrite_mount_info( stream , DRIVER_DYNAMIC_FORECAST );
    block_fs_driver_fwrite_mount_info( stream , DRIVER_DYNAMIC_ANALYZED );
    //plain_driver_index_fwrite_mount_info( stream ,                     DEFAULT_PLAIN_INDEX_PATH);
    block_fs_driver_index_fwrite_mount_info( stream );
  } else
    util_abort("%s: unrecognized driver id:%d \n",__func__ , driver_impl);
  
  __dir_offset = ftell( stream );
  {
    set_type * set = set_alloc_empty();
    set_add_key( set , default_dir );
    set_fwrite( set , stream);
    set_free( set );
  }
  util_fwrite_string( default_dir , stream );   /* Read directory  */
  util_fwrite_string( default_dir , stream );   /* Write directory */ 
  fclose(stream);
  return __dir_offset;
}
  


/**
   Storing the variables current_read_case and current_write_case.
*/

static void __fs_update_map(const char * config_file , long int __dir_offset , set_type * cases , const char * current_read_case, const char * current_write_case) {
  FILE * stream = util_fopen(config_file , "r+");
  fseek(stream , __dir_offset , SEEK_SET);  /* Skipping all the driver information - which is left untouched. */
  set_fwrite( cases , stream );
  util_fwrite_string( current_read_case , stream );
  util_fwrite_string( current_write_case , stream );
  fclose(stream);
}

/*****************************************************************/

static void enkf_fs_upgrade_100(int old_version, const char * config_file, const char * root_path , const char * new_dir) {
  char * new_path = util_alloc_filename(root_path , new_dir , NULL);
  printf("Upgrading enkf file system located in: %s\n",root_path);
  util_make_path( new_path );
  {
    DIR * dirH = opendir( root_path );
    struct dirent * dp;
    if (dirH == NULL)
      util_abort("%s: failed to open directory: %s \n",__func__ , root_path);
    
    do {
      dp = readdir(dirH);
      if (dp != NULL) {
	if (util_sscanf_int(dp->d_name , NULL)) {
	  char * old_entry = util_alloc_filename(root_path , dp->d_name , NULL);
	  char * new_entry = util_alloc_filename(new_path  , dp->d_name , NULL);

	  rename( old_entry , new_entry );
	  printf("Moving %s -> %s \n",old_entry , new_entry);
	  
	  free( old_entry );
	  free( new_entry );
	}
      } 
    } while (dp != NULL);
    closedir(dirH);
  }
  util_unlink_existing( config_file );
  enkf_fs_fwrite_new_mount_map( config_file , new_dir , PLAIN_DRIVER_ID);  /* Create new blank mount map */
}

/*****************************************************************/
/** Function for upgrading filesystem layout from version 102 -> 103. */
/* 
   This is a quite extensive upgrade, touching all the files except
   the index nodes. The upgrade involves changing from the (FILE *) based 
   reading/writing of nodes, to an approach based on (buffer_type *).
   
   This upgrade is quite extensive, there is a
   enkf_node_upgrade_file_103() function, and even special
   xxx_upgrade_103 functions for all the node types.
*/


static void upgrade_file_103(const char * root_path , const char * file , void * arg) {
  arg_pack_type * arg_pack = arg_pack_safe_cast( arg );
  int   total_files        = arg_pack_iget_int( arg_pack , 0);
  int * files              = arg_pack_iget_adress( arg_pack , 1);
  msg_type * msg           = arg_pack_iget_ptr( arg_pack , 2 );    

  enkf_impl_type impl;
  {
    char * filename          = util_alloc_filename( root_path , file , NULL );
    FILE * stream       = util_fopen( filename , "r");
    impl = util_fread_int( stream );
    fclose(stream);
    free(filename);
  }
  
  (*files) += 1;
  enkf_node_upgrade_file_103( root_path , file , impl , 100 * (*files) / total_files ,  msg);
}



static void file_count(const char * path, const char * file , void * arg) {
  arg_pack_type * arg_pack = arg_pack_safe_cast( arg );
  int   * file_count       = arg_pack_iget_adress( arg_pack , 0);
  msg_type * msg           = arg_pack_iget_ptr( arg_pack , 2 );    
  char txt[32];

  (*file_count)    += 1;
  if (((*file_count) % 100) == 0) {
    sprintf(txt , "Counting files: %d" , *file_count);
    msg_update(msg , txt);
  }
}



static void enkf_fs_upgrade_103(const char * config_file , const char * root_path) {
  printf("\n*****************************************************************\n");
  printf("** Upgrading the enkf filesystem from version 102 to 103,      **\n");
  printf("** this will take some time. There will be a .backup directory **\n");
  printf("** in every directory containing the original version 102      **\n");
  printf("** files. These directories can be removed when the upgrade has**\n");
  printf("** completed successfully.                                     **\n"),
  printf("**                                                             **\n");       
  printf("**   Hold on to your helmet :-)                                **\n");
  printf("**                                                             **\n");       
  printf("*****************************************************************\n");
  {
    msg_type * msg = msg_alloc("Upgrading: ");
    int file_count__ = 0;
    arg_pack_type * arg_pack = arg_pack_alloc();
    msg_show( msg );

    arg_pack_append_int( arg_pack , file_count__ );  
    arg_pack_append_int( arg_pack , 0 );
    arg_pack_append_ptr( arg_pack , msg );

    util_walk_directory(root_path , file_count , arg_pack);
    util_walk_directory(root_path , upgrade_file_103 , arg_pack);
    msg_free( msg , true );
    arg_pack_free( arg_pack );
  }

  /* 
     OK - have gone through all files in the filesystems - just write
     an updated mount map, with the version number and return.
  */
  {
    buffer_type * buffer = buffer_fread_alloc( config_file );
    buffer_fskip( buffer , sizeof( long ));            /* Skpping the FS_MAGIC_IC */
    buffer_fwrite_int( buffer , CURRENT_FS_VERSION );  /* inplace update of the buffer */
    buffer_store( buffer , config_file );
    buffer_free( buffer );
  }
}





/*****************************************************************/
/** Function for upgrading filesystem layout from version 101 -> 102. */
/*
  From filesystem version 102, there are no longer four different
  driver types/implementations. Instead of special drivers the special
  code for parameters and static nodes is handled at the enkf_fs layer.

  
  What this upgrade function does is:

   o In the driver records:

       DRIVER_CATEGORY   DRIVER_ID    INFO 
       int               int          void *

     The DRIVER_ID flag is rewritten to PLAIN_DRIVER_ID for all the
     drivers (i.e. plain_driver_parameter, _static, _dynamic) are
     discarded.

   o The special dynamic driver (with both forecast and analyzed) is
     discarded, and instead an extra driver is introduced (separate
     drivers for dynamic-forecast and dynamic-analyzed).

*/


static void enkf_fs_upgrade_102(const char * config_file , const char * root_path) {
  set_type * cases = set_alloc_empty();
  char * read_case = NULL;
  char * write_case = NULL;
  char * parameter_read_fmt = NULL;
  char * static_read_fmt = NULL;
  char * dynamic_analyzed_read_fmt = NULL;
  char * dynamic_forecast_read_fmt = NULL;
  char * index_read_fmt = NULL;
  
  char * parameter_write_fmt = NULL;
  char * static_write_fmt = NULL;
  char * dynamic_analyzed_write_fmt = NULL;
  char * dynamic_forecast_write_fmt = NULL;
  char * index_write_fmt = NULL;
  
  {
    char * backup_file = util_alloc_sprintf("%s.backup" , config_file);
    util_copy_file( config_file , backup_file , true);
    printf("********************************************************************\n");
    printf("** Upgrading your enkf filesystem from version 101 to version 102 **\n");
    printf("********************************************************************\n");
    printf("Backup stored in %s \n",backup_file);
    free(backup_file);
  }
  
  /* Reading in the old content. */
  {
    FILE * stream    = util_fopen(config_file , "r"); 
    util_fskip_long( stream );  /* FS tag */
    util_fskip_int( stream );   /* version id */
    int i;
    for (i=0; i < 8; i++) {
      bool read                      = util_fread_bool( stream );
      fs_driver_type driver_category = util_fread_int( stream );
      util_fskip_int( stream ); /* driver id */
      
      
      switch(driver_category) {
      case(DRIVER_PARAMETER):
	if (read)
	  parameter_read_fmt  = util_fread_alloc_string( stream );
	else
	  parameter_write_fmt = util_fread_alloc_string( stream );
	break;
      case(DRIVER_STATIC):
	if (read)
	  static_read_fmt = util_fread_alloc_string( stream );
	else
	  static_write_fmt = util_fread_alloc_string( stream );
	break;
      case(DRIVER_DYNAMIC):
	if (read) {
	  dynamic_forecast_read_fmt = util_fread_alloc_string( stream );
	  dynamic_analyzed_read_fmt = util_fread_alloc_string( stream );
	} else {
	  dynamic_forecast_write_fmt = util_fread_alloc_string( stream );
	  dynamic_analyzed_write_fmt = util_fread_alloc_string( stream );
	}
	break;
      case(DRIVER_INDEX):
	if (read)
	  index_read_fmt = util_fread_alloc_string( stream );
	else
	  index_write_fmt = util_fread_alloc_string( stream );
	break;
      default:
	util_abort("%s: fatal error driver_category:%d not recognized. \n",__func__);
      }
    }

    cases      = set_fread_alloc( stream );
    read_case  = util_fread_alloc_string( stream );
    write_case = util_fread_alloc_string( stream );
    fclose(stream);
  }
  
  /* 
     OK - now we have read in all the old information - time to write
     a new and updated map.
  */
  
  {
    FILE * stream = util_fopen( config_file , "w");
    
    util_fwrite_long(FS_MAGIC_ID , stream);
    util_fwrite_int(CURRENT_FS_VERSION , stream);
    util_fwrite_bool( true , stream);
    util_fwrite_int( DRIVER_PARAMETER , stream);
    util_fwrite_int( PLAIN_DRIVER_ID , stream);
    util_fwrite_string(  parameter_read_fmt , stream);

    util_fwrite_bool(  false , stream);
    util_fwrite_int( DRIVER_PARAMETER , stream);
    util_fwrite_int( PLAIN_DRIVER_ID , stream);
    util_fwrite_string(  parameter_write_fmt , stream);

    /*****************************************************************/

    util_fwrite_bool(  true , stream);
    util_fwrite_int( DRIVER_STATIC , stream);
    util_fwrite_int( PLAIN_DRIVER_ID , stream);
    util_fwrite_string(  static_read_fmt , stream);

    util_fwrite_bool(  false , stream);
    util_fwrite_int( DRIVER_STATIC , stream);
    util_fwrite_int( PLAIN_DRIVER_ID , stream);
    util_fwrite_string(  static_write_fmt , stream);

    /*****************************************************************/

    util_fwrite_bool(  true , stream);
    util_fwrite_int( DRIVER_DYNAMIC_FORECAST , stream);
    util_fwrite_int( PLAIN_DRIVER_ID , stream);
    util_fwrite_string(  dynamic_forecast_read_fmt , stream);

    util_fwrite_bool(  false , stream);
    util_fwrite_int( DRIVER_DYNAMIC_FORECAST , stream);
    util_fwrite_int( PLAIN_DRIVER_ID , stream);
    util_fwrite_string(  dynamic_forecast_write_fmt , stream);

    /*****************************************************************/

    util_fwrite_bool(  true , stream);
    util_fwrite_int( DRIVER_DYNAMIC_ANALYZED , stream);
    util_fwrite_int( PLAIN_DRIVER_ID , stream);
    util_fwrite_string(  dynamic_analyzed_read_fmt , stream);

    util_fwrite_bool(  false , stream);
    util_fwrite_int( DRIVER_DYNAMIC_ANALYZED , stream);
    util_fwrite_int( PLAIN_DRIVER_ID , stream);
    util_fwrite_string(  dynamic_analyzed_write_fmt , stream);
    
    /*****************************************************************/
    
    util_fwrite_bool(  true , stream);
    util_fwrite_int( DRIVER_INDEX , stream);
    util_fwrite_int( PLAIN_DRIVER_INDEX_ID , stream);
    util_fwrite_string(  index_read_fmt , stream);

    util_fwrite_bool(  false , stream);
    util_fwrite_int( DRIVER_INDEX , stream);
    util_fwrite_int( PLAIN_DRIVER_INDEX_ID , stream);
    util_fwrite_string(  index_write_fmt , stream);
    
    /*****************************************************************/
    
    set_fwrite( cases , stream );
    util_fwrite_string(read_case , stream);
    util_fwrite_string( write_case , stream);
    
    fclose(stream);
  }

  free( parameter_read_fmt );
  free( static_read_fmt );
  free( dynamic_analyzed_read_fmt );
  free( dynamic_forecast_read_fmt );
  free( index_read_fmt );

  free( parameter_write_fmt );
  free( static_write_fmt );
  free( dynamic_analyzed_write_fmt );
  free( dynamic_forecast_write_fmt );
  free( index_write_fmt );
  
  free( read_case );
  free( write_case );
}



/*****************************************************************/
/** Functions for upgrading filesystem layout from version 100 -> 101. */


void enkf_fs_upgrade_101_static( const char * static_root_path , const char * kw_path , msg_type * msg) {
  DIR * dirH = opendir( kw_path );
  if (dirH != NULL) {
    struct dirent * dp;
    do {
      dp = readdir(dirH);
      if (dp != NULL) {
	if (dp->d_name[0] != '.') {	  
	  char * new_path;
	  char * old_path = util_alloc_filename( kw_path , dp->d_name , NULL);
	  int    occurence;
	  util_sscanf_int( dp->d_name , &occurence);
	  new_path = util_alloc_sprintf("%s_%d" , kw_path , occurence);
	  rename( old_path , new_path );
	  msg_update(msg ,  old_path );
	  rmdir( kw_path );
	  free(new_path);
	  free(old_path);
	}
      }
    } while (dp != NULL);
    closedir( dirH );
  }
}




/* iterating over all static keywords. */
static void enkf_fs_upgrade_101_member( const char * root_path , msg_type * msg) {
  char * static_path = util_alloc_filename(root_path , "Static" , NULL);
  DIR * dirH = opendir( static_path );
  if (dirH != NULL) {
    struct dirent * dp;
    do {
      dp = readdir(dirH);
      if (dp != NULL) {
	char * full_path = util_alloc_filename( static_path , dp->d_name , NULL);
	if (util_is_directory(full_path)) 
	  if (dp->d_name[0] != '.') 
	    enkf_fs_upgrade_101_static( static_path , full_path , msg);
	free(full_path);
      }
    } while (dp != NULL);
    closedir( dirH );
  }
}


/* Iterating over all members */
static void enkf_fs_upgrade_101_timestep( const char * root_path , msg_type * msg) {
  DIR * dirH = opendir( root_path );
  struct dirent * dp;
  do {
    dp = readdir(dirH);
    if (dp != NULL) {
      char * full_path = util_alloc_filename( root_path , dp->d_name , NULL);
      if (util_is_directory(full_path)) 
	if (dp->d_name[0] != '.')
	  enkf_fs_upgrade_101_member( full_path , msg);
      free(full_path);
    }
  } while (dp != NULL);
  closedir( dirH );
}


/* Iterating over all timesteps */
static void enkf_fs_upgrade_101_case( const char * root_path , msg_type * msg) {
  DIR * dirH = opendir( root_path );
  struct dirent * dp;
  do {
    dp = readdir(dirH);
    if (dp != NULL) {
      char * full_path = util_alloc_filename( root_path , dp->d_name , NULL);
      if (util_is_directory(full_path)) 
	if (dp->d_name[0] != '.')
	  enkf_fs_upgrade_101_timestep( full_path , msg);
      free(full_path);
    }
  } while (dp != NULL);
  closedir( dirH );
}



static void  enkf_fs_upgrade_101_kwlist(const char * path , const char * file , void * arg) {
   if (strcmp(file , "kw_list") == 0) {
     /** This is a kw_list file which should be upgraded. */
     int i;
     char * full_path = util_alloc_filename( path , file , NULL);
     stringlist_type * kw_list;
     
     /* Reading */
     {
       FILE * stream = util_fopen( full_path , "r");
       kw_list = stringlist_fread_alloc( stream );
       fclose(stream);
     }
     
     
     /* Updating the list in place */
     for (i=0; i < stringlist_get_size( kw_list ); i++) {
       const char * kw = stringlist_iget( kw_list , i);
       bool dynamic = false;
       /* Nothing like some good old hard-coding. */
       if (strcmp(kw , "PRESSURE") == 0) dynamic = true;
       if (strcmp(kw , "SGAS")     == 0) dynamic = true;
       if (strcmp(kw , "SWAT")     == 0) dynamic = true;
       if (strcmp(kw , "RS") 	   == 0) dynamic = true;
       if (strcmp(kw , "RV") 	   == 0) dynamic = true;
       if (!dynamic) {
	 char * new_kw = util_alloc_sprintf("%s_0" , kw);
	 stringlist_iset_owned_ref( kw_list , i , new_kw);
       }
     }
     
     
    /* Writing */
     {
       FILE * stream = util_fopen( full_path , "w");
       stringlist_fwrite( kw_list , stream);
       fclose(stream);
     }
     free(full_path);
   }
}


static void enkf_fs_upgrade_101(const char * config_file, const char * root_path) {
  bool   backup = false;
  char * backup_path;
  backup_path = util_alloc_tmp_file("/tmp" , "enkf-backup" , true);

  printf("*****************************************************************************\n");
  printf("**  The filesystem for storage in the enkf has been upgraded. enkf will    **\n"); 
  printf("**  now upgrade your database, it takes some TIME, but everything should   **\n");
  printf("**  happen automatically. Observe that when enkf starts it will not        **\n");
  printf("**  remember which case you used last - apart from that everything should  **\n");
  printf("**  be unchanged.                                                          **\n");
  printf("*****************************************************************************\n");

  if (backup) {
    char * prompt = util_alloc_sprintf("Storing backup in: %s ==> " , backup_path);
    util_copy_directory( root_path , backup_path , prompt); 
    free( prompt );
  }
  free( backup_path );
  util_walk_directory( root_path , enkf_fs_upgrade_101_kwlist , NULL);
  {
    char     * current_case = NULL;
    set_type * cases        = set_alloc_empty();
    msg_type * msg 	    = msg_alloc("Upgrading: ");
    DIR * dirH     	    = opendir( root_path );
    struct dirent * dp;
    msg_show(msg);
    do {
      dp = readdir(dirH);
      if (dp != NULL) {
	char * full_path = util_alloc_filename( root_path , dp->d_name , NULL);
	if (util_is_directory(full_path)) 
	  if (dp->d_name[0] != '.') {
	    set_add_key( cases , dp->d_name );
	    enkf_fs_upgrade_101_case( full_path , msg);
	    if (current_case == NULL)
	      current_case = util_alloc_string_copy( dp->d_name );
	  }
	free( full_path );
      }
    } while (dp != NULL);
    closedir( dirH );
    msg_free(msg , true);
    {
      long int __dir_offset = enkf_fs_fwrite_new_mount_map( config_file , "DUMMY" , PLAIN_DRIVER_ID);         /* Create new blank mount map - have lost the cases.*/
      __fs_update_map( config_file , __dir_offset , cases , current_case , current_case );  /* Recovering the cases - the current is random. */
    }
    free(current_case);
    printf("Upgrading of %s is complete - continuing with normal boot process.\n" , root_path);
  }
}



static int enkf_fs_get_fs_version__(FILE * stream) {
  int version;
  long fs_tag = util_fread_long( stream );
  if (fs_tag == FS_MAGIC_ID) 
    version = util_fread_int(stream);
  else
    version = 0;
  return version;
}


/**
   -1 : No mount map found. 
    0 : Old mount map without version info.
    x : Actual version info. 
*/

static int enkf_fs_get_fs_version(const char * config_file) {
  int version = -1;
  if (util_file_exists(config_file)) {
    FILE * stream = util_fopen(config_file , "r");
    version = enkf_fs_get_fs_version__(stream);
    fclose(stream);
  } 
  return version;
}




/**
   This function upgrades to version 104 of the filesystem. In terms
   of on-disk-information this only involves the format of the mount
   map.

   The main change in 103 -> 104 is that we abondon separate driver
   for read and write, and that is essentially what happens in the
   mount map as well.
*/


static void enkf_fs_upgrade_104(const char * config_file, const char * root_path) {
  char * backup_file = util_alloc_sprintf("%s.backup" , config_file);
  {
    util_copy_file( config_file , backup_file , true);
    printf("********************************************************************\n");
    printf("** Upgrading your enkf filesystem from version 103 to version 104 **\n");
    printf("********************************************************************\n");
    printf("Backup stored in %s \n",backup_file);
  }
  {
    const int num_drivers = 10;
    FILE * read_stream    = util_fopen( backup_file , "r"); 
    FILE * write_stream   = util_fopen( config_file , "w");

    enkf_fs_get_fs_version__(read_stream);   /* Just top skip version header */
    util_fwrite_long(FS_MAGIC_ID , write_stream);
    util_fwrite_int(CURRENT_FS_VERSION , write_stream);

    for (int i=0; i < num_drivers; i++) {
      long int driver_end;
      void * driver                  = NULL;
      bool read                      = util_fread_bool( read_stream );
      long int driver_start          = ftell( read_stream );
      fs_driver_impl driver_id       = util_fread_int( read_stream );
      
      switch(driver_id) {
      case(PLAIN_DRIVER_INDEX_ID):
        driver = plain_driver_index_fread_alloc( root_path , read_stream );
        break;
      case(PLAIN_DRIVER_ID):
        driver = plain_driver_fread_alloc( root_path , read_stream );
        break;
      case(SQLITE_DRIVER_ID):
        driver = sqlite3_driver_fread_alloc( root_path , read_stream );
        break;
      default:
        util_abort("%s: fatal error in mount_map:%s - driver ID:%d not recognized. Driver nr:%d \n",__func__ , config_file , driver_id , i);
      }
      driver_end = ftell( read_stream );


      /* Copy all the 'read' drivers (without the boolean read flag) to the output stream. */
      if (read) {
        fseek( read_stream , driver_start , SEEK_SET );
        {
          int size = driver_end - driver_start;
          void * buffer = util_malloc( size , __func__);
          util_fread( buffer , 1 , size , read_stream , __func__);
          util_fwrite( buffer , 1 , size , write_stream , __func__);
          free( buffer );
        }
        fseek( read_stream , driver_end , SEEK_SET );
      }
    }
    {
      /* Copying the case information over to the new mount_map */
      char * dir = NULL;
      int num_dir = util_fread_int( read_stream );
      util_fwrite_int( num_dir , write_stream );
      if (num_dir == 0) 
        util_abort("%s: must have a directory ... \n",__func__);
      

      /* Loading the set of directories. */
      for (int i=0; i < num_dir; i++) {
        dir = util_fread_realloc_string( dir , read_stream); 
        util_fwrite_string( dir , write_stream );
      }
      
      /* The currently selected cases. */
      /* Read dir */
      dir = util_fread_realloc_string( dir , read_stream ); 
      util_fwrite_string( dir , write_stream );
      
      /* write dir */
      dir = util_fread_realloc_string( dir , read_stream ); 
      util_fwrite_string( dir , write_stream );
      free(dir);
      
    }
    fclose( read_stream );
    fclose( write_stream );
  }
}







static void enkf_fs_update_map(const enkf_fs_type * fs) {
  __fs_update_map( fs->mount_map , fs->__dir_offset , fs->dir_set , fs->current_read_dir , fs->current_write_dir);
}



/*****************************************************************/

void enkf_fs_add_dir(enkf_fs_type * fs, const char * dir) {
  set_add_key(fs->dir_set , dir);
  enkf_fs_update_map(fs);
}

bool enkf_fs_has_dir(const enkf_fs_type * fs, const char * dir) {
  return set_has_key( fs->dir_set , dir );
}


bool enkf_fs_select_dir__(enkf_fs_type * fs, const char * dir, bool read , bool auto_mkdir) {
  bool has_dir = true;
  if (dir != NULL) {
    if (!set_has_key(fs->dir_set , dir)) {
      if ((!read) && (auto_mkdir)) {
	enkf_fs_add_dir(fs , dir);
	has_dir = false;
      } else {
	fprintf(stderr,"%s: fatal error - can not select directory: \"%s\" \n",__func__ , dir);
	fprintf(stderr,"Available: directories: ");
	set_fprintf(fs->dir_set , stderr);
	fprintf(stderr,"\n");
	
	has_dir = false;
	util_abort("%s: Aborting (make the directory first ...) \n",__func__);
      }
    }
  }


  if (read) 
    fs->current_read_dir = util_realloc_string_copy( fs->current_read_dir , dir );
  else
    fs->current_write_dir = util_realloc_string_copy( fs->current_write_dir , dir );


  fs->dynamic_forecast->select_dir(fs->dynamic_forecast , dir , read);
  fs->dynamic_analyzed->select_dir(fs->dynamic_analyzed , dir , read);
  fs->parameter->select_dir(fs->parameter , dir , read);
  fs->index->select_dir(fs->index , dir , read);    
  fs->eclipse_static->select_dir(fs->eclipse_static , dir , read);
  
  
  enkf_fs_update_map(fs);
  return has_dir;
}


void enkf_fs_select_read_dir(enkf_fs_type * fs, const char * dir) {
  if ((fs->current_read_dir == NULL) || (dir == NULL) || (strcmp(fs->current_read_dir , dir) != 0)) {
    enkf_fs_select_dir__(fs , dir , true , false);
  }
}


/**
   If auto_mkdir == true a directory which does not exist will be
   created; if auto_mkdir == false the function will util_abort() if
   the directory does not exist.

   Return true if the directory already exists, and false if the
   directory was created (latter return value only applies when the
   function is called with auto_mkdir == true).
*/

bool enkf_fs_select_write_dir(enkf_fs_type * fs, const char * dir , bool auto_mkdir) {
  if ((fs->current_write_dir == NULL) || (dir == NULL) || (strcmp(fs->current_write_dir , dir) != 0)) {
    return enkf_fs_select_dir__(fs , dir , false , auto_mkdir);
  }
  return false;
}




enkf_fs_type * enkf_fs_mount(const char * root_path , fs_driver_impl driver_impl, const char *mount_info , const char * lock_path) {
  const bool   use_locking = false;
  const char * default_dir = DEFAULT_CASE;
  char * config_file       = util_alloc_filename(root_path , mount_info , NULL);  /* This file should be protected - at all costs. */
  int    version           = enkf_fs_get_fs_version( config_file );
  
  
  if (version > CURRENT_FS_VERSION)
    util_exit("The ensemble at %s has FILE_SYSTEM_VERSION:%d  -  the current enkf FILE_SYSTEM_VERSION is:%d - UPGRADE enkf \n", root_path , version , CURRENT_FS_VERSION);
  
  util_make_path(root_path);                                    	    	  /* Creating root directory */
  if (version == -1)
    enkf_fs_fwrite_new_mount_map( config_file , default_dir ,driver_impl);  	  /* Create blank mount map */
  else if (version < CURRENT_FS_VERSION) {  /* Upgrade file system layout */
    
    if (version == 0) {
      enkf_fs_upgrade_100(version , config_file , root_path , default_dir);       /* Upgrade disk information - and create new mount map. */
      version = 100;
    }

    
    if (version == 100) {
      enkf_fs_upgrade_101( config_file , root_path );
      version = 101;
    }
    

    if (version == 101) {
      enkf_fs_upgrade_102( config_file , root_path );
      version = 102;
    } 

    if (version == 102) {
      enkf_fs_upgrade_103( config_file , root_path );
      version = 103;
    }

    if (version == 103) {
      enkf_fs_upgrade_104( config_file , root_path );
      version = 104;
    }
  }

  version = enkf_fs_get_fs_version( config_file );
  if (version != CURRENT_FS_VERSION)
    util_abort("%s: upgrading of filesystem rooted in:%s failed. \n",root_path);

  
  /** Reading the mount map */
  {
    const int num_drivers      = 5;
    enkf_fs_type * fs          = util_malloc(sizeof * fs , __func__);

    fs->use_locking            = use_locking;
    fs->lockfile               = NULL;

    fs->index                  = NULL;
    fs->eclipse_static         = NULL;
    fs->parameter              = NULL;
    fs->dynamic_forecast       = NULL;
    fs->dynamic_analyzed       = NULL;

    fs->dir_set             = set_alloc_empty();
    fs->current_read_dir    = NULL; 
    fs->current_write_dir   = NULL;
    {
      fs->mount_map      = util_alloc_string_copy( config_file );
      FILE * stream      = util_fopen(fs->mount_map , "r");
      int i;
      
      enkf_fs_get_fs_version__(stream);   /* Just top skip version header */
      for (i=0; i < num_drivers; i++) {
	void * driver   = NULL;
	fs_driver_type driver_category = util_fread_int( stream );
	fs_driver_impl driver_id       = util_fread_int( stream );

	switch(driver_id) {
	case(PLAIN_DRIVER_INDEX_ID):
	  driver = plain_driver_index_fread_alloc( root_path , stream );
	  break;
	case(PLAIN_DRIVER_ID):
	  driver = plain_driver_fread_alloc( root_path , stream );
	  break;
	case(SQLITE_DRIVER_ID):
	  driver = sqlite3_driver_fread_alloc( root_path , stream );
	  break;
	case(BLOCK_FS_DRIVER_ID):
	  driver = block_fs_driver_fread_alloc( root_path , stream );
	  break;
        case(BLOCK_FS_DRIVER_INDEX_ID):
	  driver = block_fs_driver_index_fread_alloc( root_path , stream );
	  break;
	default:
	  util_abort("%s: fatal error in mount_map:%s - driver ID:%d not recognized. Driver nr:%d \n",__func__ , fs->mount_map , driver_id , i);
	}
	
	switch(driver_category) {
	case(DRIVER_PARAMETER):
          fs->parameter  = driver;
	  break;
	case(DRIVER_STATIC):
	  fs->eclipse_static = driver;
	  break;
	case(DRIVER_DYNAMIC_FORECAST):
	  fs->dynamic_forecast = driver;
	  break;
	case(DRIVER_DYNAMIC_ANALYZED):
	  fs->dynamic_analyzed = driver;
	  break;
	case(DRIVER_INDEX):
	  fs->index = driver;
	  break;
	default:
	  util_abort("%s: fatal error in mount_map:%s - driver category:%d not recognized \n",__func__ , fs->mount_map , driver_category);
	}
      }
      
      if (fs->parameter	        == NULL) util_abort("%s: fatal error - mount map in:%s did not contain parameter driver.\n", __func__ , fs->mount_map);
      if (fs->eclipse_static    == NULL) util_abort("%s: fatal error - mount map in:%s did not contain ecl_static driver.\n",__func__ , fs->mount_map);
      if (fs->index             == NULL) util_abort("%s: fatal error - mount map in:%s did not contain index driver.\n",     __func__ , fs->mount_map);
      if (fs->dynamic_forecast  == NULL) util_abort("%s: fatal error - mount map in:%s did not contain dynamic forecast driver.\n",     __func__ , fs->mount_map);
      if (fs->dynamic_analyzed  == NULL) util_abort("%s: fatal error - mount map in:%s did not contain dynamic analyzed driver.\n",     __func__ , fs->mount_map);
      
      /* Starting on the directory information. */
      fs->__dir_offset = ftell(stream);
      {
	char * dir = NULL;
	int num_dir = util_fread_int( stream );
	if (num_dir == 0) 
	  util_abort("%s: must have a directory ... \n",__func__);
        
	/* Loading the set of directories. */
	for (int i=0; i < num_dir; i++) {
	  dir = util_fread_realloc_string( dir , stream); 
	  enkf_fs_add_dir( fs , dir );
	}
        free(dir);
        
        /* Loading and selecting the currently selected read and write directories. */
        {
          char * current_read_dir  = util_fread_alloc_string( stream );
          char * current_write_dir = util_fread_alloc_string( stream );
          
          enkf_fs_select_read_dir(fs , current_read_dir);
          enkf_fs_select_write_dir(fs , current_write_dir , false);
        }
      }
      fclose( stream );
    }
  

    basic_driver_assert_cast(fs->dynamic_analyzed);
    basic_driver_assert_cast(fs->dynamic_forecast);
    basic_driver_assert_cast(fs->eclipse_static);
    basic_driver_assert_cast(fs->parameter);
    basic_driver_index_assert_cast(fs->index);
    
    if (use_locking) {
      {
	char * ens_path    = util_alloc_string_copy(root_path);
	util_string_tr( ens_path , UTIL_PATH_SEP_CHAR , '_' );
	fs->lockfile = util_alloc_filename(lock_path , ens_path , "ensemble_lock");
	free(ens_path);
      }
      
      
      if (util_try_lockf(fs->lockfile , S_IWUSR + S_IWGRP, &fs->lock_fd))
	fs->read_only = false;
      else {
	fs->read_only = true;
	fprintf(stderr,"------------------------------------------------------------------------\n");
	fprintf(stderr,"| Warning: another EnKF instance has currently locked the ensemble at\n| \'%s\' for writing - this instance will be read-only.\n",root_path);
	fprintf(stderr,"-------------------------------------------------------------------------\n");
      }
    } else
      fs->read_only = false;
    
    free( config_file );
    return fs;
  }
}




static void enkf_fs_free_driver(basic_driver_type * driver) {
  driver->free_driver(driver);
}

static void enkf_fs_free_index_driver(basic_driver_index_type * driver) {
  driver->free_driver(driver);
}



void enkf_fs_free(enkf_fs_type * fs) {
  enkf_fs_free_driver(fs->dynamic_forecast);
  enkf_fs_free_driver(fs->dynamic_analyzed);
  enkf_fs_free_driver(fs->parameter);
  enkf_fs_free_driver(fs->eclipse_static);
  enkf_fs_free_index_driver(fs->index);
  
  util_safe_free(fs->current_read_dir);
  util_safe_free(fs->current_write_dir);
  free(fs->mount_map);
  set_free(fs->dir_set);
  if (fs->use_locking) {
    close(fs->lock_fd);
    util_unlink_existing(fs->lockfile);
  }
  util_safe_free(fs->lockfile);
  free(fs);
}


static void * select_dynamic_driver(enkf_fs_type * fs , state_enum state ) {
  void * driver = NULL;

  if (state == ANALYZED) 
    driver = fs->dynamic_analyzed;
  else if (state == FORECAST) 
    driver = fs->dynamic_forecast;
  else
    util_abort("%s: fatal internal error \n",__func__);
  
  return driver;
}


static void * enkf_fs_select_driver(enkf_fs_type * fs , enkf_var_type var_type, state_enum state , const char * key) {
  void * driver = NULL;
  switch (var_type) {
  case(PARAMETER):
    driver = fs->parameter;
    break;
  case(DYNAMIC_RESULT):
    driver = select_dynamic_driver( fs , state );
    break;
  case(DYNAMIC_STATE):
    driver = select_dynamic_driver( fs , state );
    break;
  case(STATIC_STATE):
    driver = fs->eclipse_static;
    break;
  default:
    util_abort("%s: fatal internal error - could not determine enkf_fs driver for object:%s[integer type:%d] - aborting.\n",__func__, key , var_type);
  }
  return driver;
}



/*****************************************************************/
/* Exported functions for enkf_node instances . */

void enkf_fs_fwrite_node(enkf_fs_type * enkf_fs , enkf_node_type * enkf_node , int report_step , int iens , state_enum state) {
  if (enkf_fs->read_only)
    util_abort("%s: attempt to write to read_only filesystem - aborting. \n",__func__);
  {
    enkf_var_type var_type = enkf_node_get_var_type(enkf_node);
    void * _driver = enkf_fs_select_driver(enkf_fs , var_type , state , enkf_node_get_key(enkf_node) );
    {
      basic_driver_type * driver = basic_driver_safe_cast(_driver);
      if (report_step == 0) {
	enkf_impl_type impl_type = enkf_node_get_impl_type(enkf_node);
	if (impl_type == SUMMARY) return;    /* For report step == 0 the summary data is just garbage. */
      }
      {
	bool   internal_state = true;
	bool   data_written;
	buffer_type * buffer = buffer_alloc(100);
	buffer_fwrite_time_t( buffer , time(NULL));
	data_written = enkf_node_store(enkf_node , buffer , internal_state , report_step , iens , state); 
	if (data_written) {
          const enkf_config_node_type * config_node = enkf_node_get_config( enkf_node );
	  driver->save(driver , config_node , report_step , iens , buffer);
	}
	buffer_free( buffer );
      }
    }
  }
}


/**
  For parameters the state is uniquely identified by the report step,
  corresponding to the __analyzed__ state. If you really want the
  forecast that is achieved by subtracting one.
*/

static int __get_parameter_report_step( int report_step , state_enum state) {
  if (state == FORECAST) {
    if (report_step > 0) /* Time step zero is special - we do not differentiate between forecast and analyzed. */
      report_step--;
  } else if (state != ANALYZED)
    util_abort("%s: asked for state:%d - internal error \n",__func__ , state);

  return report_step;
}

pthread_mutex_t read_mutex = PTHREAD_MUTEX_INITIALIZER;

void enkf_fs_fread_node(enkf_fs_type * enkf_fs , enkf_node_type * enkf_node , int report_step , int iens , state_enum state) {
  const enkf_config_node_type * config_node = enkf_node_get_config( enkf_node );
  enkf_var_type var_type     = enkf_config_node_get_var_type(config_node);
  basic_driver_type * driver = enkf_fs_select_driver(enkf_fs , var_type , state , enkf_node_get_key(enkf_node) );
  
  if (var_type == PARAMETER) {
    report_step = __get_parameter_report_step( report_step , state );
    
    /*
      Observe that if we do not find the filename we are looking for, we
      seek backwards through the report numbers, all the way back to
      report_nr 0. The direct motivation for this functionality is the
      following situation:
      
        1. We do a spin-up from report 0 to report R1.
      
        2. We start the assimulation from R1, then we have to go all the
           way back to report 0 to get hold of the parameter.
    */
    while (!driver->has_node( driver , config_node , report_step , iens )) {
      report_step--;
      if (report_step < 0)
	util_abort("%s: can not find any stored item for key:%s(%d). Forgot to initialize ensemble ??? \n",__func__ , enkf_node_get_key( enkf_node ) , iens);
    }
  }

  {
    buffer_type * buffer = buffer_alloc(100);
    driver->load(driver , config_node ,  report_step , iens , buffer);
    buffer_fskip_time_t( buffer );
    enkf_node_load(enkf_node , buffer , report_step, iens , state);
    buffer_free( buffer );
  }
}


bool enkf_fs_has_node(enkf_fs_type * enkf_fs , const enkf_config_node_type * config_node , int report_step , int iens , state_enum state) {
  enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
  {
    basic_driver_type * driver = basic_driver_safe_cast(enkf_fs_select_driver(enkf_fs , var_type , state , enkf_config_node_get_key( config_node ) ));
    return driver->has_node(driver , config_node , report_step , iens ); 
  }
}


enkf_node_type * enkf_fs_fread_alloc_node(enkf_fs_type * enkf_fs , const enkf_config_node_type * config_node , int report_step , int iens, state_enum state) {
  enkf_node_type * node = enkf_node_alloc(config_node);
  enkf_fs_fread_node(enkf_fs , node , report_step , iens , state);
  return node;
}



void enkf_fs_copy_node(enkf_fs_type * enkf_fs, 
		       enkf_config_node_type * config_node, 
		       int report_step_from, int iens_from, state_enum state_from, /* src state */
		       int report_step_to  , int iens_to  , state_enum state_to) { /* target state */

  enkf_node_type * enkf_node = enkf_fs_fread_alloc_node(enkf_fs, config_node, report_step_from, iens_from, state_from);
  enkf_fs_fwrite_node(enkf_fs, enkf_node, report_step_to, iens_to, state_to);
  enkf_node_free(enkf_node);

}


/**
  Copy an ensemble of nodes. Note that the limits are inclusive.
*/
void enkf_fs_copy_ensemble(enkf_fs_type * enkf_fs, 
			   enkf_config_node_type * config_node,               
			   int report_step_from, state_enum state_from,  /* src state */
			   int report_step_to  , state_enum state_to,      /* target state */
			   int ens_size, 
			   const int * __permutations) {
  
  int * permutations;
  if (__permutations == NULL) {
    /* No reordering of ensemble members. */
    permutations = util_malloc( ens_size * sizeof * permutations , __func__);
    for (int i = 0; i < ens_size; i++)
      permutations[i] = i;
  } else
    permutations = (int *) __permutations;

  
  for(int iens_from =0; iens_from < ens_size; iens_from++) {
    int iens_to = permutations[iens_from];
    enkf_fs_copy_node(enkf_fs, config_node, report_step_from, iens_from , state_from, report_step_to, iens_to , state_to);
  }

  if (__permutations == NULL) 
    free( permutations );
}



/**
  Scatter a node. Note that the limits are inclusive.
*/
void enkf_fs_scatter_node(enkf_fs_type * enkf_fs, enkf_config_node_type * config_node, int report_step, int iens_source, state_enum state, int iens1, int iens2)
{
  enkf_node_type * enkf_node = enkf_fs_fread_alloc_node(enkf_fs, config_node, report_step, iens_source, state);
  for(int i = iens1; i <= iens2; i++)
    enkf_fs_fwrite_node(enkf_fs, enkf_node, report_step, i, state);
  enkf_node_free(enkf_node);
}


/**
   This function will load a node from the filesystem if it is
   available; if not it will just return false.

   The state argument can be 'both' - in which case it will first try
   the analyzed, and then subsequently the forecast before giving up
   and returning false. If the function returns true with state ==
   'both' it is no way to determine which version was actually loaded.
*/

bool enkf_fs_try_fread_node(enkf_fs_type * enkf_fs , enkf_node_type * node , int report_step , int iens , state_enum state) {
  if (state == BOTH) {
    bool analyzed_return = enkf_fs_try_fread_node(enkf_fs , node , report_step , iens , ANALYZED);
    if (analyzed_return)
      return true;
    else
      return enkf_fs_try_fread_node(enkf_fs , node , report_step , iens , FORECAST);
  } else {

    if (enkf_fs_has_node(enkf_fs , enkf_node_get_config(node) , report_step , iens , state)) {
      enkf_fs_fread_node(enkf_fs , node , report_step , iens , state);
      return true;
    } else
      return false;
  }

}


/**
   Will hav NULL values if the member can not be found.
*/

enkf_node_type ** enkf_fs_fread_alloc_ensemble( enkf_fs_type * fs , const enkf_config_node_type * config_node , int report_step , int iens1 , int iens2 , state_enum state) {
  enkf_node_type ** ensemble = util_malloc( (iens2 - iens1) * sizeof * ensemble , __func__);
  for (int iens = iens1; iens < iens2; iens++) {
    state_enum load_state;
    ensemble[iens - iens1] = NULL;

    if (state == BOTH) {
      if (enkf_fs_has_node( fs , config_node , report_step , iens , ANALYZED))
        load_state = ANALYZED;
      else if (enkf_fs_has_node( fs , config_node , report_step , iens , FORECAST))
        load_state = FORECAST;
      else
        load_state = UNDEFINED;
    } else 
      load_state = state;

    if (load_state != UNDEFINED)
      ensemble[iens - iens1] = enkf_fs_fread_alloc_node(fs , config_node , report_step , iens , load_state);
  }
  
  return ensemble;
}



/*****************************************************************/

void enkf_fs_interactive_select_directory(void * arg) {
  arg_pack_type * arg_pack         = arg_pack_safe_cast( arg );
  enkf_fs_type   * fs              = arg_pack_iget_ptr(arg_pack , 0);  
  bool read                        = arg_pack_iget_bool(arg_pack , 1); 
  menu_item_type * item            = arg_pack_iget_ptr(arg_pack , 2);

  char dir[256];
  if (read)
    printf("Give directory for reading ==> ");
  else
    printf("Give directory for writing ==> ");
  scanf("%s" , dir);
  enkf_fs_select_dir__(fs , dir , read , true);
  
  {
    char * menu_label;
    if (read)
      menu_label = util_alloc_sprintf("Set new directory for reading:%s" , dir);
    else
      menu_label = util_alloc_sprintf("Set new directory for writing:%s" , dir);

    menu_item_set_label( item , menu_label );
    free(menu_label);
  }
}

/**
   Checks if the current_read_dir == current_write_dir.
*/

bool enkf_fs_rw_equal(const enkf_fs_type * fs) {
  if (strcmp(fs->current_read_dir, fs->current_write_dir) == 0)
    return true;
  else
    return false;
}


const char * enkf_fs_get_write_dir(const enkf_fs_type * fs) { return fs->current_write_dir; }

const char * enkf_fs_get_read_dir(const enkf_fs_type * fs) {
  return fs->current_read_dir;
}


stringlist_type * enkf_fs_alloc_dirlist(const enkf_fs_type * fs) {
  stringlist_type * dirlist = stringlist_alloc_new();
  int           num_keys = set_get_size(fs->dir_set); 
  char ** keylist  = set_alloc_keylist(fs->dir_set);
  for (int i=0; i<num_keys; i++)
    stringlist_append_owned_ref(dirlist, keylist[i]);
  return dirlist;
}


/*****************************************************************/
/* Index related functions  . */

void enkf_fs_fwrite_restart_kw_list(enkf_fs_type * enkf_fs , int report_step , int iens, const stringlist_type * kw_list) {
  basic_driver_index_type * index = enkf_fs->index;
  buffer_type * buffer = buffer_alloc(1024);
  stringlist_buffer_fwrite( kw_list , buffer );
  index->save_kwlist( index , report_step , iens , buffer );
  buffer_free( buffer );
}



void enkf_fs_fread_restart_kw_list(enkf_fs_type * enkf_fs , int report_step , int iens, stringlist_type * kw_list) {
  basic_driver_index_type * index = enkf_fs->index;
  buffer_type * buffer = buffer_alloc(1024);
  index->load_kwlist( index , report_step , iens , buffer  );
  stringlist_buffer_fread( kw_list , buffer );
  buffer_free( buffer );
}




