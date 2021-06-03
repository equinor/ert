/*
   Copyright (C) 2011  Equinor ASA, Norway.

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
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

#include <ert/util/util.h>
#include <ert/util/type_macros.h>

#include <ert/res_util/path_fmt.hpp>
#include <ert/res_util/res_log.hpp>

#include <ert/enkf/block_fs_driver.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/misfit_ensemble.hpp>

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
  is not necessary to implement these functions, the enkf_fs layer has simple
  functions doing repeated calls to the enkf_fs_fread_node() function.


  The drivers
  -----------

  The enkf_fs layer does not self implement the functions to read and write
  nodes. Instead what happens is:

    1. We determine the type of the node (static/dynamic/parameter), and select
       the appropriate driver.

    2. The appropriate driver is called to implement e.g. the fread_node
       functions.

  The different types of data have different characteristics, which the driver is
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
  Like the parameter driver this also only stores one version of the data,
  however in addition it has to query the node for an ID to support multiple
  occurring keywords in ECLIPSE restart files.

  Currently only the plain_driver_xxx family has been implemented. Observe that
  there is no dependencies between the drivers, it is perfectly possible to
  implement a new driver for storage of static data only. (There is probably a
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
       indefinitely.

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
  fs_driver_type (fs_types.h) and DRIVER_ID is one of the integer
  values in fs_driver_impl. The last void * data is whatever
  (serialized) info the driver needs to bootstrap. This info is
  written by the drivers xxxx_fwrite_mount_info() function, and it is
  used when the driver is allocated with xxxx_fread_alloc().

  The different drivers can be in arbitrary order in the
  enkf_mount_info file, but when four records are read it checks that
  all drivers have been initialized, and aborts if that is not the
  case.

  If the enkf_mount_info file is deleted that can cause problems.
  It is currently 'protected' with chomd a-w - but that is of course not
  foolprof.
*/




/**
   Observe the following convention: the initial ensemble at report
   step 0 is supposed to be analyzed. If we ask for the forecast at
   report_step 0, we should get the analyzed value.
*/


#define ENKF_FS_TYPE_ID           1089763
#define ENKF_MOUNT_MAP            "enkf_mount_info"
#define SUMMARY_KEY_SET_FILE      "summary-key-set"
#define TIME_MAP_FILE             "time-map"
#define STATE_MAP_FILE            "state-map"
#define MISFIT_ENSEMBLE_FILE      "misfit-ensemble"
#define CASE_CONFIG_FILE          "case_config"

struct enkf_fs_struct {
  UTIL_TYPE_ID_DECLARATION;
  char                   * case_name;
  char                   * root_path;
  char                   * mount_point;    // mount_point = root_path / case_name; the mount_point is the fundamental INPUT.

  char                   * lock_file;
  int                      lock_fd;

  fs_driver_type         * dynamic_forecast;
  fs_driver_type         * parameter;
  fs_driver_type         * index ;

  bool                        read_only;             /* Whether this filesystem has been mounted read-only. */
  time_map_type             * time_map;
  cases_config_type         * cases_config;
  state_map_type            * state_map;
  summary_key_set_type      * summary_key_set;
  misfit_ensemble_type      * misfit_ensemble;
  /*
     The variables below here are for storing arbitrary files within
     the enkf_fs storage directory, but not as serialized enkf_nodes.
  */
  path_fmt_type             * case_fmt;
  path_fmt_type             * case_member_fmt;
  path_fmt_type             * case_tstep_fmt;
  path_fmt_type             * case_tstep_member_fmt;

  int                         refcount;
  int                         runcount;  // Counts the number of simulations currently writing to this enkf_fs; the purpose is to
                                         // be able to answer the question: Is this case currently 'running'?
};


/*****************************************************************/


UTIL_SAFE_CAST_FUNCTION( enkf_fs , ENKF_FS_TYPE_ID)
UTIL_IS_INSTANCE_FUNCTION( enkf_fs , ENKF_FS_TYPE_ID)

static void enkf_fs_umount( enkf_fs_type * fs );

int enkf_fs_incref( enkf_fs_type * fs ) {
  fs->refcount++;

  res_log_fdebug("Calling incref on: %s . Refcount after incref:%d", fs->mount_point, fs->refcount);

  return fs->refcount;
}


int enkf_fs_decref( enkf_fs_type * fs ) {
  int refcount;
  fs->refcount--;
  refcount = fs->refcount;

  if (fs->refcount < 0)
    util_abort("%s: Internal inconsistency in file system. The filesystem refcount:%d is < 0 \n",__func__ , fs->refcount);

  res_log_fdebug("Calling decref on: %s . Refcount after decref:%d", fs->mount_point, fs->refcount);
  if (refcount == 0)
    enkf_fs_umount( fs );

  return refcount;
}


int enkf_fs_get_refcount( const enkf_fs_type * fs ) {
  return fs->refcount;
}


enkf_fs_type * enkf_fs_get_ref( enkf_fs_type * fs ) {
  enkf_fs_incref( fs );
  return fs;
}


static enkf_fs_type * enkf_fs_alloc_empty( const char * mount_point ) {
  enkf_fs_type * fs = (enkf_fs_type *)util_malloc(sizeof * fs );
  UTIL_TYPE_ID_INIT( fs , ENKF_FS_TYPE_ID );
  fs->time_map               = time_map_alloc(  );
  fs->cases_config           = cases_config_alloc();
  fs->state_map              = state_map_alloc();
  fs->summary_key_set        = summary_key_set_alloc();
  fs->misfit_ensemble        = misfit_ensemble_alloc();
  fs->index                  = NULL;
  fs->parameter              = NULL;
  fs->dynamic_forecast       = NULL;
  fs->read_only              = true;
  fs->mount_point            = util_alloc_string_copy( mount_point );
  fs->refcount               = 0;
  fs->runcount               = 0;
  fs->lock_fd                = 0;

  if (mount_point == NULL)
    util_abort("%s: fatal internal error: mount_point == NULL \n",__func__);
  {
    char ** path_tmp;
    int     path_len;

    util_path_split( fs->mount_point , &path_len , &path_tmp);
    fs->case_name = util_alloc_string_copy( path_tmp[path_len - 1]);
    fs->root_path = util_alloc_joined_string( (const char **) path_tmp , path_len , UTIL_PATH_SEP_STRING);
    fs->lock_file = util_alloc_filename( fs->mount_point , fs->case_name , "lock");

    if (util_try_lockf( fs->lock_file , S_IWUSR + S_IWGRP , &fs->lock_fd)) {
      fs->read_only = false;
    } else {
      fprintf(stderr," Another program has already opened filesystem read-write - this instance will be UNSYNCRONIZED read-only. Cross your fingers ....\n");
      fs->read_only = true;
    }

    util_free_stringlist( path_tmp , path_len );
  }
  return fs;
}




static int enkf_fs_fread_fs_version__(FILE * stream) {
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
   0  : Old mount map without version info.
   x  : Actual version info.
*/

static int enkf_fs_get_fs_version__(const char * config_file) {
  int version = -1;
  if (util_file_exists(config_file)) {
    FILE * stream = util_fopen(config_file , "r");
    version = enkf_fs_fread_fs_version__(stream);
    fclose(stream);
  }
  return version;
}

/**
   Function written to look for old (version <= 104) mount info maps.
*/

int enkf_fs_get_version104( const char * path ) {
  char * config_file = util_alloc_filename( path , ENKF_MOUNT_MAP, NULL);
  int version = enkf_fs_get_fs_version__( config_file );
  free( config_file );
  return version;
}







/*****************************************************************/


static void enkf_fs_init_path_fmt( enkf_fs_type * fs) {
  /*
    Installing the path_fmt instances for the storage of arbitrary files.
  */
  fs->case_fmt              = path_fmt_alloc_directory_fmt( DEFAULT_CASE_PATH );
  fs->case_member_fmt       = path_fmt_alloc_directory_fmt( DEFAULT_CASE_MEMBER_PATH );
  fs->case_tstep_fmt        = path_fmt_alloc_directory_fmt( DEFAULT_CASE_TSTEP_PATH );
  fs->case_tstep_member_fmt = path_fmt_alloc_directory_fmt( DEFAULT_CASE_TSTEP_MEMBER_PATH );

}

static void enkf_fs_create_block_fs( FILE * stream , int num_drivers , const char * mount_point , void * arg) {

  block_fs_driver_create_fs( stream , mount_point , DRIVER_PARAMETER        , num_drivers , "Ensemble/mod_%d" , "PARAMETER");
  block_fs_driver_create_fs( stream , mount_point , DRIVER_DYNAMIC_FORECAST , num_drivers , "Ensemble/mod_%d" , "FORECAST");
  block_fs_driver_create_fs( stream , mount_point , DRIVER_INDEX            , 1           , "Index"           , "INDEX");

}


static void enkf_fs_assign_driver( enkf_fs_type * fs , fs_driver_type * driver , fs_driver_enum driver_type ) {
  switch(driver_type) {
  case(DRIVER_PARAMETER):
    fs->parameter = driver;
    break;
  case(DRIVER_DYNAMIC_FORECAST):
    fs->dynamic_forecast = driver;
    break;
  case(DRIVER_INDEX):
    fs->index = driver;
    break;
  case(DRIVER_STATIC):
    util_abort("%s: internal error - should not assign a STATIC driver \n",__func__);
    break;
  case(DRIVER_DYNAMIC_ANALYZED):
    util_abort("%s: internal error - should not assign a DYNAMIC_ANALYZED driver \n",__func__);
    break;
  }
}


static enkf_fs_type *  enkf_fs_mount_block_fs( FILE * fstab_stream , const char * mount_point  ) {
  enkf_fs_type * fs = enkf_fs_alloc_empty( mount_point );

  {
    while (true) {
      fs_driver_enum driver_type;
      if (fread( &driver_type , sizeof driver_type , 1 , fstab_stream) == 1) {
        if (fs_types_valid( driver_type )) {
          fs_driver_type * driver = (fs_driver_type * ) block_fs_driver_open( fstab_stream , mount_point , driver_type , fs->read_only);
          enkf_fs_assign_driver( fs , driver , driver_type );
        } else
          block_fs_driver_fskip( fstab_stream );
      } else
        break;
    }
  }

  return fs;
}

enkf_fs_type * enkf_fs_create_fs( const char * mount_point, fs_driver_impl driver_id , void * arg , bool mount) {
  const int num_drivers = 32;
  FILE * stream = fs_driver_open_fstab( mount_point , true );
  if (stream != NULL) {
    fs_driver_init_fstab( stream, driver_id);
    {
      switch( driver_id ) {
      case( BLOCK_FS_DRIVER_ID ):
        enkf_fs_create_block_fs( stream , num_drivers , mount_point , arg );
        break;
      default:
        util_abort("%s: Invalid driver_id value:%d \n",__func__ , driver_id );
      }
    }
    fclose( stream );
  }

  if (mount)
    return enkf_fs_mount( mount_point );
  else
    return NULL;
}

static void enkf_fs_fsync_time_map( enkf_fs_type * fs ) {
  char * filename = enkf_fs_alloc_case_filename( fs , TIME_MAP_FILE );
  time_map_fwrite( fs->time_map , filename );
  free( filename );
}


static void enkf_fs_fread_time_map( enkf_fs_type * fs ) {
  char * filename = enkf_fs_alloc_case_filename( fs , TIME_MAP_FILE );
  time_map_fread( fs->time_map , filename );
  free( filename );
}


static void enkf_fs_fsync_cases_config( enkf_fs_type * fs ) {
  char * filename = enkf_fs_alloc_case_filename( fs , CASE_CONFIG_FILE );
  cases_config_fwrite( fs->cases_config , filename );
  free( filename );
}

static void enkf_fs_fsync_state_map( enkf_fs_type * fs ) {
  char * filename = enkf_fs_alloc_case_filename( fs , STATE_MAP_FILE );
  state_map_fwrite( fs->state_map , filename );
  free( filename );
}

static void enkf_fs_fsync_summary_key_set( enkf_fs_type * fs ) {
  char * filename = enkf_fs_alloc_case_filename( fs , SUMMARY_KEY_SET_FILE );
  summary_key_set_fwrite( fs->summary_key_set , filename );
  free( filename );
}

static void enkf_fs_fread_cases_config( enkf_fs_type * fs ) {
  char * filename = enkf_fs_alloc_case_filename( fs , CASE_CONFIG_FILE );
  cases_config_fread( fs->cases_config , filename );
  free( filename );
}


static void enkf_fs_fread_state_map( enkf_fs_type * fs ) {
  char * filename = enkf_fs_alloc_case_filename( fs , STATE_MAP_FILE );
  state_map_fread( fs->state_map , filename );
  free( filename );
}

static void enkf_fs_fread_summary_key_set( enkf_fs_type * fs ) {
  char * filename = enkf_fs_alloc_case_filename( fs , SUMMARY_KEY_SET_FILE );
  summary_key_set_fread( fs->summary_key_set , filename );
  free( filename );
}

state_map_type * enkf_fs_alloc_readonly_state_map( const char * mount_point ) {
  path_fmt_type * path_fmt = path_fmt_alloc_directory_fmt( DEFAULT_CASE_PATH );
  char * filename = path_fmt_alloc_file( path_fmt , false , mount_point , STATE_MAP_FILE);

  state_map_type * state_map = state_map_fread_alloc_readonly( filename );

  path_fmt_free( path_fmt );
  free( filename );
  return state_map;
}

time_map_type * enkf_fs_alloc_readonly_time_map( const char * mount_point ) {
  path_fmt_type * path_fmt = path_fmt_alloc_directory_fmt( DEFAULT_CASE_PATH );
  char * filename = path_fmt_alloc_file( path_fmt , false , mount_point , TIME_MAP_FILE);

  time_map_type * time_map = time_map_fread_alloc_readonly( filename );

  path_fmt_free( path_fmt );
  free( filename );
  return time_map;
}



static void enkf_fs_fread_misfit( enkf_fs_type * fs ) {
  FILE * stream = enkf_fs_open_excase_file( fs , MISFIT_ENSEMBLE_FILE );
  if (stream != NULL) {
    misfit_ensemble_fread( fs->misfit_ensemble , stream );
    fclose( stream );
  }
}


static void enkf_fs_fwrite_misfit( enkf_fs_type * fs ) {
  if (misfit_ensemble_initialized( fs->misfit_ensemble )) {
    FILE * stream = enkf_fs_open_case_file( fs , MISFIT_ENSEMBLE_FILE , "w");
    misfit_ensemble_fwrite( fs->misfit_ensemble , stream );
    fclose( stream );
  }
}



int enkf_fs_disk_version(const char * mount_point ) {
  int disk_version = -1;
  FILE * stream = fs_driver_open_fstab( mount_point , false );
  if (stream) {
    disk_version = fs_driver_fread_version( stream );
    fclose( stream );
  }
  return disk_version;
}


bool enkf_fs_update_disk_version(const char * mount_point , int src_version , int target_version) {
  if (enkf_fs_disk_version( mount_point ) == src_version) {
    char * fstab_file = fs_driver_alloc_fstab_file( mount_point );
    FILE * stream = util_fopen( fstab_file , "r+");

    fseek( stream , 0L, SEEK_SET );
    fs_driver_assert_magic( stream );
    util_fwrite_int( target_version , stream );

    fclose( stream );
    free( fstab_file );

    return true;
  }  else
    return false;
}


enkf_fs_type * enkf_fs_mount(const char * mount_point) {
  FILE * stream = fs_driver_open_fstab(mount_point, false);

  if (!stream)
    return NULL;

  enkf_fs_type * fs = NULL;
  fs_driver_assert_magic(stream);
  fs_driver_assert_version(stream, mount_point);

  fs_driver_impl driver_id = (fs_driver_impl) util_fread_int(stream);

  switch(driver_id) {
  case(BLOCK_FS_DRIVER_ID):
    fs = enkf_fs_mount_block_fs(stream, mount_point);
    res_log_fdebug("Mounting (block_fs) point %s.", mount_point);
    break;
  default:
    util_abort("%s: unrecognized driver_id:%d \n", __func__, driver_id);
  }

  fclose(stream);
  enkf_fs_init_path_fmt(fs);
  enkf_fs_fread_time_map(fs);
  enkf_fs_fread_cases_config(fs);
  enkf_fs_fread_state_map(fs);
  enkf_fs_fread_summary_key_set(fs);
  enkf_fs_fread_misfit(fs);

  enkf_fs_get_ref(fs);
  return fs;
}


bool enkf_fs_exists( const char * mount_point ) {
  bool exists   = false;

  FILE * stream = fs_driver_open_fstab( mount_point , false );
  if (stream != NULL) {
    exists = true;
    fclose( stream );
  }

  return exists;
}




/*****************************************************************/



static void enkf_fs_free_driver(fs_driver_type * driver) {
  driver->free_driver(driver);
}


static void enkf_fs_umount(enkf_fs_type * fs) {
  if (!fs->read_only) {
    enkf_fs_fsync(fs);
    enkf_fs_fwrite_misfit(fs);
  }

  int refcount = fs->refcount;
  if (refcount > 0)
    util_abort("%s: Internal inconsistency - "
               "tried to umount a filesystem with refcount:%d\n",
               __func__, refcount);

  res_log_fdebug("%s umount filesystem %s", __func__, fs->mount_point);

  enkf_fs_free_driver(fs->dynamic_forecast);
  enkf_fs_free_driver(fs->parameter);
  enkf_fs_free_driver(fs->index);

  if (fs->lock_fd > 0) {
    close(fs->lock_fd);  // Closing the lock_file file descriptor - and releasing the lock.
    util_unlink_existing(fs->lock_file);
  }

  free(fs->case_name);
  free(fs->root_path);
  free(fs->lock_file);
  free(fs->mount_point);
  path_fmt_free(fs->case_fmt);
  path_fmt_free(fs->case_member_fmt);
  path_fmt_free(fs->case_tstep_fmt);
  path_fmt_free(fs->case_tstep_member_fmt);

  state_map_free(fs->state_map);
  summary_key_set_free(fs->summary_key_set);
  time_map_free(fs->time_map);
  cases_config_free(fs->cases_config);
  misfit_ensemble_free(fs->misfit_ensemble);
  free(fs);
}





static void * enkf_fs_select_driver(enkf_fs_type * fs , enkf_var_type var_type, const char * key) {
  void * driver = NULL;
  switch (var_type) {
  case(DYNAMIC_RESULT):
    driver = fs->dynamic_forecast;
    break;
  case(EXT_PARAMETER):
    driver = fs->parameter;
    break;
  case(PARAMETER):
    driver = fs->parameter;
    break;
  default:
    util_abort("%s: fatal internal error - could not determine enkf_fs driver for object:%s[integer type:%d] - aborting.\n",__func__, key , var_type);
  }
  return driver;
}



/*****************************************************************/
/* Exported functions for enkf_node instances . */


static void enkf_fs_fsync_driver( fs_driver_type * driver ) {
  if (driver->fsync_driver != NULL)
    driver->fsync_driver( driver );
}



void enkf_fs_fsync( enkf_fs_type * fs ) {
  enkf_fs_fsync_driver( fs->parameter );
  enkf_fs_fsync_driver( fs->dynamic_forecast );
  enkf_fs_fsync_driver( fs->index );

  enkf_fs_fsync_time_map( fs );
  enkf_fs_fsync_cases_config( fs) ;
  enkf_fs_fsync_state_map( fs );
  enkf_fs_fsync_summary_key_set( fs );
}


void enkf_fs_fread_node(enkf_fs_type * enkf_fs , buffer_type * buffer ,
                        const char * node_key ,
                        enkf_var_type var_type ,
                        int report_step,
                        int iens) {

  fs_driver_type * driver = (fs_driver_type * ) enkf_fs_select_driver(enkf_fs , var_type , node_key );
  if (var_type == PARAMETER)
    /* Parameters are *ONLY* stored at report_step == 0 */
    report_step = 0;

  buffer_rewind( buffer );
  driver->load_node(driver , node_key ,  report_step , iens , buffer);
}


void enkf_fs_fread_vector(enkf_fs_type * enkf_fs , buffer_type * buffer ,
                          const char * node_key ,
                          enkf_var_type var_type ,
                          int iens) {

  fs_driver_type * driver = (fs_driver_type * ) enkf_fs_select_driver(enkf_fs , var_type , node_key );

  buffer_rewind( buffer );
  driver->load_vector(driver , node_key ,  iens , buffer);
}



bool enkf_fs_has_node(enkf_fs_type * enkf_fs , const char * node_key , enkf_var_type var_type , int report_step , int iens) {
  fs_driver_type * driver = fs_driver_safe_cast(enkf_fs_select_driver(enkf_fs , var_type , node_key));
  return driver->has_node(driver , node_key , report_step , iens );
}


bool enkf_fs_has_vector(enkf_fs_type * enkf_fs , const char * node_key , enkf_var_type var_type , int iens ) {
  fs_driver_type * driver = fs_driver_safe_cast(enkf_fs_select_driver(enkf_fs , var_type ,  node_key));
  return driver->has_vector(driver , node_key , iens );
}

void enkf_fs_fwrite_node(enkf_fs_type * enkf_fs , buffer_type * buffer , const char * node_key, enkf_var_type var_type,
                         int report_step , int iens ) {
  if (enkf_fs->read_only)
    util_abort("%s: attempt to write to read_only filesystem mounted at:%s - aborting. \n",__func__ , enkf_fs->mount_point);


  if ((var_type == PARAMETER) && (report_step > 0))
    util_abort("%s: Parameters can only be saved for report_step = 0   %s:%d\n", __func__ , node_key , report_step);
  {
    void * _driver = enkf_fs_select_driver(enkf_fs , var_type , node_key);
    {
      fs_driver_type * driver = fs_driver_safe_cast(_driver);
      driver->save_node(driver , node_key , report_step , iens , buffer);
    }
  }
}


void enkf_fs_fwrite_vector(enkf_fs_type * enkf_fs , buffer_type * buffer , const char * node_key, enkf_var_type var_type,
                           int iens ) {
  if (enkf_fs->read_only)
    util_abort("%s: attempt to write to read_only filesystem mounted at:%s - aborting. \n",__func__ , enkf_fs->mount_point);
  {
    void * _driver = enkf_fs_select_driver(enkf_fs , var_type , node_key);
    {
      fs_driver_type * driver = fs_driver_safe_cast(_driver);
      driver->save_vector(driver , node_key  , iens , buffer);
    }
  }
}




/*****************************************************************/





/*****************************************************************/

const char * enkf_fs_get_mount_point( const enkf_fs_type * fs ) {
  return fs->mount_point;
}

const char * enkf_fs_get_case_name( const enkf_fs_type * fs ) {
  return fs->case_name;
}


bool enkf_fs_is_read_only(const enkf_fs_type * fs) {
    return fs->read_only;
}

/*****************************************************************/
/* write_dir / read_dir confusion. */

char * enkf_fs_alloc_case_filename( const enkf_fs_type * fs , const char * input_name) {
  char * filename = path_fmt_alloc_file( fs->case_fmt , false , fs->mount_point , input_name);
  return filename;
}

char * enkf_fs_alloc_case_tstep_filename( const enkf_fs_type * fs , int tstep , const char * input_name) {
  char * filename    = path_fmt_alloc_file( fs->case_tstep_fmt , false , fs->mount_point , tstep , input_name);
  return filename;
}

char * enkf_fs_alloc_case_tstep_member_filename( const enkf_fs_type * fs , int tstep , int iens , const char * input_name) {
  char * filename = path_fmt_alloc_file( fs->case_tstep_member_fmt , false , fs->mount_point , tstep , iens , input_name);
  return filename;
}



FILE * enkf_fs_open_case_file( const enkf_fs_type * fs , const char * input_name , const char * mode) {
  char * filename = enkf_fs_alloc_case_filename( fs , input_name );
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


FILE * enkf_fs_open_excase_tstep_file( const enkf_fs_type * fs , const char * input_name , int tstep ) {
  char * filename = enkf_fs_alloc_case_tstep_filename( fs , tstep , input_name );
  FILE * stream   = enkf_fs_open_exfile( filename );
  free( filename );
  return stream;
}


/*****************************************************************/

time_map_type * enkf_fs_get_time_map( const enkf_fs_type * fs ) {
  return fs->time_map;
}

cases_config_type * enkf_fs_get_cases_config( const enkf_fs_type * fs) {
  return fs->cases_config;
}

state_map_type * enkf_fs_get_state_map( const enkf_fs_type * fs ) {
  return fs->state_map;
}

summary_key_set_type * enkf_fs_get_summary_key_set( const enkf_fs_type * fs ) {
  return fs->summary_key_set;
}

misfit_ensemble_type * enkf_fs_get_misfit_ensemble( const enkf_fs_type * fs ) {
  return fs->misfit_ensemble;
}

void enkf_fs_increase_run_count(enkf_fs_type * fs) {
  fs->runcount = fs->runcount + 1;
}

void enkf_fs_decrease_run_count(enkf_fs_type * fs) {
  fs->runcount = fs->runcount - 1;
}

bool enkf_fs_is_running(const enkf_fs_type * fs) {
  return (fs->runcount > 0);
}
