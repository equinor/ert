/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'enkf_fs.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/

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
#include <fs_driver.h>
#include <dirent.h>
#include <fs_types.h>
#include <menu.h>
#include <arg_pack.h>
#include <ecl_static_kw.h>
#include <plain_driver_index.h>
#include <plain_driver.h>
#include <block_fs_driver.h>
#include <block_fs_driver_index.h>
#include <stringlist.h>
#include <arg_pack.h>
#include <pthread.h>
#include <gen_data.h>

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
                                  |   2190: started to distribute ert binary internally
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

   The block_fs driver is added - and reasonably stabilized in this
   version.


   Observe that all the upgrade functions were removed at svn:3305.
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
  enkf_mount_info file, but when four records are read it checks that
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


#define ENKF_FS_TYPE_ID 1089763

struct enkf_fs_struct {
  UTIL_TYPE_ID_DECLARATION;
  char                      * root_path;
  fs_driver_type         * dynamic_forecast;
  fs_driver_type         * dynamic_analyzed;
  fs_driver_type         * parameter;
  fs_driver_type         * eclipse_static;
  fs_driver_index_type   * index ;

  bool                        read_only;             /* Whether this filesystem has been mounted read-only. */
  set_type                  * dir_set;               /* Set containing the existing directories. */
  char                      * current_read_dir;      /* The currently active "directory" for reading. */
  char                      * current_write_dir;     /* The currently active directory fro writing. */ 
  char                      * mount_map;             /* Binary file containing all the information the filesystem needs to bootstrap itself. Updated when e.g. new directories are added. */
  long int                     __dir_offset;         /* The offset into the mount map where the directory information starts - very internal. */
  /*****************************************************************/
  /* 
     The variables below here are for storing arbitrary files within 
     the enkf_fs storage directory, but not as serialized enkf_nodes.
  */
  path_fmt_type             * case_fmt;
  path_fmt_type             * case_member_fmt;
  path_fmt_type             * case_tstep_fmt;
  path_fmt_type             * case_tstep_member_fmt;
};

/*****************************************************************/


UTIL_SAFE_CAST_FUNCTION( enkf_fs , ENKF_FS_TYPE_ID)
UTIL_IS_INSTANCE_FUNCTION( enkf_fs , ENKF_FS_TYPE_ID)


/** 
    This function writes a new default mount map; to be called if no
    mount map is found. Returns the __dir_offset.
*/

long int enkf_fs_fwrite_new_mount_map(const char * mount_map, const char * default_dir, fs_driver_impl driver_impl) {
  const int num_block_fs_drivers = ENKF_DEFAULT_NUM_BLOCK_FS_DRIVERS;
  long int  __dir_offset;
  FILE * stream = util_fopen( mount_map , "w");
  util_fwrite_long(FS_MAGIC_ID , stream);
  util_fwrite_int(CURRENT_FS_VERSION , stream);

  if (driver_impl == PLAIN_DRIVER_ID) {
    /* Writing the mount map for brand new PLAIN - fs. */
    plain_driver_fwrite_mount_info( stream , DRIVER_PARAMETER        , DEFAULT_PLAIN_PARAMETER_PATH);
    plain_driver_fwrite_mount_info( stream , DRIVER_STATIC           , DEFAULT_PLAIN_STATIC_PATH);
    plain_driver_fwrite_mount_info( stream , DRIVER_DYNAMIC_FORECAST , DEFAULT_PLAIN_DYNAMIC_FORECAST_PATH);
    plain_driver_fwrite_mount_info( stream , DRIVER_DYNAMIC_ANALYZED , DEFAULT_PLAIN_DYNAMIC_ANALYZED_PATH);
    plain_driver_index_fwrite_mount_info( stream ,                     DEFAULT_PLAIN_INDEX_PATH);
  } else if (driver_impl == BLOCK_FS_DRIVER_ID) {
    block_fs_driver_fwrite_mount_info( stream , DRIVER_PARAMETER        , num_block_fs_drivers);
    block_fs_driver_fwrite_mount_info( stream , DRIVER_STATIC           , num_block_fs_drivers);
    block_fs_driver_fwrite_mount_info( stream , DRIVER_DYNAMIC_FORECAST , num_block_fs_drivers);
    block_fs_driver_fwrite_mount_info( stream , DRIVER_DYNAMIC_ANALYZED , num_block_fs_drivers);
    //plain_driver_index_fwrite_mount_info( stream ,                     DEFAULT_PLAIN_INDEX_PATH);
    block_fs_driver_index_fwrite_mount_info( stream );
  } else
    util_abort("%s: unrecognized driver id:%d \n",__func__ , driver_impl);


  /**
     Observe that function - writing the initial default/plain mount
     map does not use the enkf_fs functions. The (one) reason to do it
     like this is that one avoids temporarily writing invalid files.
  */
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
  /**
     Have (for a long time) had a bug in this which has stored mount
     maps with NULL for the current cases. The broken files have
     typically been updated immediately with correct files, but
     nevertheless there has been a short window with broken
     configurations on disk. Extra nazi checking this now.
  */

  if (current_read_case == NULL)
    util_abort("%s: internal bug - trying to store NULL for read_case.\n",__func__);
  
  if (current_write_case == NULL)
    util_abort("%s: internal bug - trying to store NULL for write_case.\n",__func__);
  
  {
    FILE * stream = util_fopen(config_file , "r+");
    fseek(stream , __dir_offset , SEEK_SET);  /* Skipping all the driver information - which is left untouched. */
    set_fwrite( cases , stream );
    
    util_fwrite_string( current_read_case , stream );
    util_fwrite_string( current_write_case , stream );
    fclose(stream);
  }
}

/*****************************************************************/







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





static void enkf_fs_update_map(const enkf_fs_type * fs) {
  __fs_update_map( fs->mount_map , fs->__dir_offset , fs->dir_set , fs->current_read_dir , fs->current_write_dir);
}



/*****************************************************************/

static void enkf_fs_add_dir__(enkf_fs_type * fs, const char * dir, bool store) {
  set_add_key(fs->dir_set , dir);
  if (store) enkf_fs_update_map(fs);
}


void enkf_fs_add_dir(enkf_fs_type * fs, const char * dir) {
  enkf_fs_add_dir__( fs , dir , true );
}





bool enkf_fs_has_dir(const enkf_fs_type * fs, const char * dir) {
  return set_has_key( fs->dir_set , dir );
}


static void enkf_fs_select_dir(enkf_fs_type * fs, const char * dir, bool read , bool read_only , bool store_map) {
  if (read) 
    fs->current_read_dir = util_realloc_string_copy( fs->current_read_dir , dir );
  else
    fs->current_write_dir = util_realloc_string_copy( fs->current_write_dir , dir );

  fs->dynamic_forecast->select_dir(fs->dynamic_forecast , dir , read , read_only);
  fs->dynamic_analyzed->select_dir(fs->dynamic_analyzed , dir , read, read_only);
  fs->parameter->select_dir(fs->parameter , dir , read , read_only);
  fs->eclipse_static->select_dir(fs->eclipse_static , dir , read, read_only);
  fs->index->select_dir(fs->index , dir , read , read_only);    
  
  if (store_map)
    enkf_fs_update_map(fs);
}



void enkf_fs_select_read_dir(enkf_fs_type * fs, const char * dir, bool update_map) {
  bool read_only = false;
  if (!util_string_equal(fs->current_read_dir , dir)) {
    if (set_has_key( fs->dir_set , dir)) 
      /* If the current_read_dir == NULL this is part of the mount process, and no need to write a (not) updated mount map. */
      enkf_fs_select_dir(fs , dir , true , read_only , (update_map && (fs->current_read_dir != NULL)));   
    else {
      /* 
         To avoid util_abort() on not existing dir the calling scope
         should check existence prior to calling this function. 
      */
      fprintf(stderr,"%s: fatal error - can not select directory: \"%s\" \n",__func__ , dir);
      fprintf(stderr,"Available: directories: ");
      set_fprintf(fs->dir_set , " " , stderr);
      fprintf(stderr,"\n");
      
      util_abort("%s: Aborting (make the directory first ...) \n",__func__);
    }
  }
}




/**
   If auto_mkdir == true a directory which does not exist will be
   created; if auto_mkdir == false the function will util_abort() if
   the directory does not exist.
*/

void enkf_fs_select_write_dir(enkf_fs_type * fs, const char * dir , bool auto_mkdir , bool update_map) {
  bool read_only = false;
  if (!util_string_equal(fs->current_write_dir , dir)) {
    if (!set_has_key( fs->dir_set , dir))
      if (auto_mkdir)
        enkf_fs_add_dir__( fs , dir , false); /* Add a dir instance - without storing a new mount map. */
    
    if (set_has_key( fs->dir_set , dir))
      enkf_fs_select_dir(fs , dir , false , read_only , (update_map && (fs->current_write_dir != NULL)));
    else {
      fprintf(stderr,"%s: fatal error - can not select directory: \"%s\" \n",__func__ , dir);
      fprintf(stderr,"Available: directories: ");
      set_fprintf(fs->dir_set , " " , stderr);
      fprintf(stderr,"\n");
      
      util_abort("%s: Aborting (make the directory first ...) \n",__func__);
    }
  }
}



/**
   The filesystem will start by selecting the case given by the input
   parameter @select_case; this input can be NULL (in which case the
   current default case is selected), if @select_case points to a
   non-existing case a warning will be printed on stderr on no
   selection will be performed.
   
   Iff the file-system does not exist; and select_case != NULL
   select_case will also be used as the default case.
*/


enkf_fs_type * enkf_fs_mount(const char * root_path , fs_driver_impl driver_impl, const char *mount_info , const char * select_case , bool update_map , bool read_only) {
  const char * default_dir = DEFAULT_CASE;
  char * config_file       = util_alloc_filename(root_path , mount_info , NULL);  /* This file should be protected - at all costs. */
  int    version           = enkf_fs_get_fs_version( config_file );
  
  
  if (version > CURRENT_FS_VERSION)
    util_exit("The ensemble at %s has FILE_SYSTEM_VERSION:%d  -  the current enkf FILE_SYSTEM_VERSION is:%d - UPGRADE enkf \n", root_path , version , CURRENT_FS_VERSION);
  
  util_make_path(root_path);                                                      /* Creating root directory */
  if (version == -1) {
    if (select_case != NULL)
      default_dir = select_case;
    enkf_fs_fwrite_new_mount_map( config_file , default_dir ,driver_impl);        /* Create blank mount map */
  } else if (version < CURRENT_FS_VERSION) {  /* Upgrade file system layout */
    /* All upgrade code was removed at svn:3305 10.th of january 2011. */
    fprintf(stderr,"Hmmm - tried to upgrade a on-disk filesystem. Use an ert version svn <= 3304 \n");
  }

  version = enkf_fs_get_fs_version( config_file );
  if (version != CURRENT_FS_VERSION)
    util_abort("%s: upgrading of filesystem rooted in:%s failed. \n",root_path);

  
  /** Reading the mount map */
  {
    const int num_drivers      = 5;
    enkf_fs_type * fs          = util_malloc(sizeof * fs , __func__);
    UTIL_TYPE_ID_INIT( fs , ENKF_FS_TYPE_ID );
    fs->root_path              = util_alloc_string_copy( root_path );
    fs->index                  = NULL;
    fs->eclipse_static         = NULL;
    fs->parameter              = NULL;
    fs->dynamic_forecast       = NULL;
    fs->dynamic_analyzed       = NULL;

    fs->dir_set             = set_alloc_empty();
    fs->current_read_dir    = NULL; 
    fs->current_write_dir   = NULL;
    fs->read_only           = read_only;
    {
      fs->mount_map      = util_alloc_string_copy( config_file );
      FILE * stream      = util_fopen(fs->mount_map , "r");
      bool store_map     = false; 
      int i;
      
      enkf_fs_get_fs_version__(stream);   /* Just top skip version header */
      for (i=0; i < num_drivers; i++) {
        void * driver   = NULL;
        fs_driver_enum driver_category = util_fread_int( stream );
        fs_driver_impl driver_id       = util_fread_int( stream );

        switch(driver_id) {
        case(PLAIN_DRIVER_INDEX_ID):
          driver = plain_driver_index_fread_alloc( root_path , stream );
          break;
        case(PLAIN_DRIVER_ID):
          driver = plain_driver_fread_alloc( root_path , stream );
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
      
      if (fs->parameter         == NULL) util_abort("%s: fatal error - mount map in:%s did not contain parameter driver.\n", __func__ , fs->mount_map);
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
          enkf_fs_add_dir__( fs , dir , false);
        }
        free(dir);
        
        
        /* Loading and selecting the currently selected read and write directories. */
        {
          char * current_read_dir  = util_fread_alloc_string( stream );
          char * current_write_dir = util_fread_alloc_string( stream );
          bool   case_selected     = false;
          
          /** 
              If the user has given an initial case for selection wr
              try to select that first; otherwise we select the
              current directories which are (hopefully) found in the
              mount map.
          */
          if (select_case != NULL) {
            if (enkf_fs_has_dir( fs , select_case )) {
              enkf_fs_select_read_dir( fs , select_case , update_map);
              enkf_fs_select_write_dir( fs , select_case , false , update_map);
              case_selected = true;
            } else
              fprintf(stderr,"** Warning: case:%s does not exist \n", select_case );
          } 

          
          /* 
             No initial case selected - try for the current selections
             in the mount map.
          */
          if (!case_selected) {
            if ((current_read_dir == NULL) || (current_write_dir == NULL)) {
              fprintf(stderr," *************************************************************\n");
              fprintf(stderr," ** Hmmmm - the mount map has been corrupted. I have lost   **\n"); 
              fprintf(stderr," ** track of your currently selected read/write cases. Will **\n");
              fprintf(stderr," ** just select the default case.                           **\n"); 
              fprintf(stderr," *************************************************************\n");
              
              enkf_fs_select_read_dir(fs , "default" , true);
              enkf_fs_select_write_dir(fs , "default" , false , true);
              store_map = true;
            } else {
              enkf_fs_select_read_dir(fs , current_read_dir , true);
              enkf_fs_select_write_dir(fs , current_write_dir , false , true);
            }
          }
          
          util_safe_free( current_read_dir );
          util_safe_free( current_write_dir );
        }
      }
      fclose( stream );
      if (store_map) /* The dir info has been updated - and we dump and updated map to file. */
        enkf_fs_update_map( fs );
    }
    

    fs_driver_assert_cast(fs->dynamic_analyzed);
    fs_driver_assert_cast(fs->dynamic_forecast);
    fs_driver_assert_cast(fs->eclipse_static);
    fs_driver_assert_cast(fs->parameter);
    fs_driver_index_assert_cast(fs->index);
    
    free( config_file );
    {
      /*
        Installing the path_fmt instances for the storage of arbitrary files.
      */
      fs->case_fmt              = path_fmt_alloc_directory_fmt( DEFAULT_CASE_PATH );
      fs->case_member_fmt       = path_fmt_alloc_directory_fmt( DEFAULT_CASE_MEMBER_PATH );
      fs->case_tstep_fmt        = path_fmt_alloc_directory_fmt( DEFAULT_CASE_TSTEP_PATH );
      fs->case_tstep_member_fmt = path_fmt_alloc_directory_fmt( DEFAULT_CASE_TSTEP_MEMBER_PATH );
    }

    return fs;
  }
}




static void enkf_fs_free_driver(fs_driver_type * driver) {
  driver->free_driver(driver);
}

static void enkf_fs_free_index_driver(fs_driver_index_type * driver) {
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
  util_safe_free(fs->root_path);
  path_fmt_free( fs->case_fmt );
  path_fmt_free( fs->case_member_fmt );
  path_fmt_free( fs->case_tstep_fmt );
  path_fmt_free( fs->case_tstep_member_fmt );
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
      fs_driver_type * driver = fs_driver_safe_cast(_driver);
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


static void enkf_fs_fsync_driver( fs_driver_type * driver ) {
  if (driver->fsync_driver != NULL)
    driver->fsync_driver( driver );
}


static void enkf_fs_fsync_driver_index( fs_driver_index_type * driver ) {
  if (driver->fsync_driver != NULL)
    driver->fsync_driver( driver );
}




void enkf_fs_fsync( enkf_fs_type * fs ) {
  {
    enkf_fs_fsync_driver( fs->parameter );
    enkf_fs_fsync_driver( fs->eclipse_static );
    enkf_fs_fsync_driver( fs->dynamic_forecast );
    enkf_fs_fsync_driver( fs->dynamic_analyzed );
    enkf_fs_fsync_driver_index( fs->index );
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

void enkf_fs_fread_node(enkf_fs_type * enkf_fs , enkf_node_type * enkf_node , int report_step , int iens , state_enum state) {
  const enkf_config_node_type * config_node = enkf_node_get_config( enkf_node );
  enkf_var_type var_type     = enkf_config_node_get_var_type(config_node);
  fs_driver_type * driver = enkf_fs_select_driver(enkf_fs , var_type , state , enkf_node_get_key(enkf_node) );
  int internal_report_step   = report_step;
  
  if (var_type == PARAMETER) {
    internal_report_step = __get_parameter_report_step( report_step , state );
    
    /*
      Observe that if we do not find the filename we are looking for, we
      seek backwards through the report numbers, all the way back to
      report_nr 0. The direct motivation for this functionality is the
      following situation:
      
        1. We do a spin-up from report 0 to report R1.
      
        2. We start the assimulation from R1, then we have to go all the
           way back to report 0 to get hold of the parameter.

    */
    while (!driver->has_node( driver , config_node , internal_report_step , iens )) {
      internal_report_step--;
      if (internal_report_step < 0)
        util_abort("%s: can not find any stored item for key:%s(%d). Forgot to initialize ensemble ??? \n",__func__ , enkf_node_get_key( enkf_node ) , iens);
    }
  }

  {
    buffer_type * buffer = buffer_alloc(100);
    driver->load(driver , config_node ,  internal_report_step , iens , buffer);
    buffer_fskip_time_t( buffer );
    enkf_node_load(enkf_node , buffer , report_step, iens , state);
    buffer_free( buffer );
  }
}


bool enkf_fs_has_node(enkf_fs_type * enkf_fs , const enkf_config_node_type * config_node , int report_step , int iens , state_enum state) {
  enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
  {
    fs_driver_type * driver = fs_driver_safe_cast(enkf_fs_select_driver(enkf_fs , var_type , state , enkf_config_node_get_key( config_node ) ));
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

  
  
  /* Hack to ensure that size is set for the gen_data instances.
     This sneeks low level stuff into a high level scope. BAD. */
  {
    enkf_impl_type impl_type = enkf_node_get_impl_type( enkf_node );
    if (impl_type == GEN_DATA) {
      /* Read the size at report_step_from */
      gen_data_type * gen_data = enkf_node_value_ptr( enkf_node );
      int size                 = gen_data_get_size( gen_data );
      
      /* Enforce the size at report_step_to */
      gen_data_assert_size( gen_data , size , report_step_to);
    }
  }
  enkf_fs_fwrite_node(enkf_fs, enkf_node, report_step_to, iens_to, state_to);
  enkf_node_free(enkf_node);
}


/**
  Copy an ensemble of nodes. Note that the limits are inclusive.
*/
void enkf_fs_copy_node_ensemble(enkf_fs_type * enkf_fs, 
                                enkf_config_node_type * config_node,               
                                int report_step_from, state_enum state_from,    /* src state */
                                int report_step_to  , state_enum state_to,      /* target state */
                                int ens_size, 
                                const int * permutations) {
  
  for(int iens_from =0; iens_from < ens_size; iens_from++) {
    int iens_to = permutations[iens_from];
    enkf_fs_copy_node(enkf_fs, config_node, report_step_from, iens_from , state_from, report_step_to, iens_to , state_to);
  }
  
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
  fs_driver_index_type * index = enkf_fs->index;
  buffer_type * buffer = buffer_alloc(1024);
  stringlist_buffer_fwrite( kw_list , buffer );
  index->save_kwlist( index , report_step , iens , buffer );
  buffer_free( buffer );
}



void enkf_fs_fread_restart_kw_list(enkf_fs_type * enkf_fs , int report_step , int iens, stringlist_type * kw_list) {
  fs_driver_index_type * index = enkf_fs->index;
  buffer_type * buffer = buffer_alloc(1024);
  index->load_kwlist( index , report_step , iens , buffer  );
  stringlist_buffer_fread( kw_list , buffer );
  buffer_free( buffer );
}


/*****************************************************************/
/*****************************************************************/
/* write_dir / read_dir confusion. */

char * enkf_fs_alloc_case_filename( const enkf_fs_type * fs , const char * input_name) {
  return path_fmt_alloc_file( fs->case_fmt , false , fs->root_path , fs->current_write_dir , input_name);
}

char * enkf_fs_alloc_case_member_filename( const enkf_fs_type * fs , int iens , const char * input_name) {
  return path_fmt_alloc_file( fs->case_member_fmt , false , fs->root_path , fs->current_write_dir , iens , input_name);
}

char * enkf_fs_alloc_case_tstep_filename( const enkf_fs_type * fs , int tstep , const char * input_name) {
  return path_fmt_alloc_file( fs->case_tstep_fmt , false , fs->root_path , fs->current_write_dir , tstep , input_name);
}

char * enkf_fs_alloc_case_tstep_member_filename( const enkf_fs_type * fs , int tstep , int iens , const char * input_name) {
  return path_fmt_alloc_file( fs->case_tstep_member_fmt , false , fs->root_path , fs->current_write_dir , tstep , iens , input_name);
}



FILE * enkf_fs_open_case_file( const enkf_fs_type * fs , const char * input_name , const char * mode) {
  char * filename = enkf_fs_alloc_case_filename( fs , input_name );
  FILE * stream   = util_mkdir_fopen( filename , mode );
  free( filename );
  return stream;
}


FILE * enkf_fs_open_case_member_file( const enkf_fs_type * fs , const char * input_name , int iens , const char * mode) {
  char * filename = enkf_fs_alloc_case_member_filename( fs , iens , input_name );
  FILE * stream   = util_mkdir_fopen( filename , mode );
  free( filename );
  return stream;
}
  

FILE * enkf_fs_open_case_tstep_file( const enkf_fs_type * fs , const char * input_name , int tstep , const char * mode) {
  char * filename = enkf_fs_alloc_case_tstep_filename( fs , tstep , input_name );
  FILE * stream   = util_mkdir_fopen( filename , mode );
  free( filename );
  return stream;
}


FILE * enkf_fs_open_case_tstep_member_file( const enkf_fs_type * fs , const char * input_name , int tstep , int iens , const char * mode) {
  char * filename = enkf_fs_alloc_case_tstep_member_filename( fs , tstep , iens , input_name );
  FILE * stream   = util_mkdir_fopen( filename , mode );
  free( filename );
  return stream;
}

/*****************************************************************/
/* 
   The open_exXXX functions will return NULL if the file does not
   already exist. These functions can only be used to open with 'r'
   mode.
*/



static FILE * enkf_fs_open_exfile( const char * filename ) {
  if (util_file_exists(filename))
    return util_fopen( filename , "r");
  else
    return NULL;
}

FILE * enkf_fs_open_excase_file( const enkf_fs_type * fs , const char * input_name ) {
  char * filename = enkf_fs_alloc_case_filename( fs , input_name );
  FILE * stream   = enkf_fs_open_exfile( filename );
  free( filename );
  return stream;
}


FILE * enkf_fs_open_excase_member_file( const enkf_fs_type * fs , const char * input_name , int iens ) {
  char * filename = enkf_fs_alloc_case_member_filename( fs , iens , input_name );
  FILE * stream   = enkf_fs_open_exfile( filename );
  free( filename );
  return stream;
}
  

FILE * enkf_fs_open_excase_tstep_file( const enkf_fs_type * fs , const char * input_name , int tstep ) {
  char * filename = enkf_fs_alloc_case_tstep_filename( fs , tstep , input_name );
  FILE * stream   = enkf_fs_open_exfile( filename );
  printf("Looking for:%s \n",filename);
  free( filename );
  return stream;
}


FILE * enkf_fs_open_excase_tstep_member_file( const enkf_fs_type * fs , const char * input_name , int tstep , int iens ) {
  char * filename = enkf_fs_alloc_case_tstep_member_filename( fs , tstep , iens , input_name );
  FILE * stream   = enkf_fs_open_exfile( filename );
  free( filename );
  return stream;
}
  

